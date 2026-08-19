"""QLoRA supervised fine-tuning of Qwen2.5-1.5B-Instruct on the mixed dataset.

WHAT QLoRA IS
-------------
Two ideas stacked:

  Quantization  The base model's frozen weights are stored in 4 bits instead of
                16. A 1.5B model drops from ~3 GB to ~1 GB. The weights are
                dequantized on the fly for each matmul, so quality loss is small.

  LoRA          Every targeted weight matrix W is frozen, and a trainable
                bypass is added beside it:  h = Wx + (alpha/r) * B(Ax)
                where A is (r x d) and B is (d x r) with r=16 << d. We train
                roughly 1% of the parameters, and the saved artifact is a ~40 MB
                adapter rather than a 3 GB model.

                B is initialized to ZERO, so BA = 0 at step 0 and the model
                starts out behaving EXACTLY like the base model. Training moves
                away from a known-good starting point, never from a random one.

WHY THESE HYPERPARAMETERS
-------------------------
lr=2e-4     ~10x a typical full-fine-tune learning rate. Legitimate here because
            only the small adapter matrices move; the base model cannot be
            damaged by a large step.
epochs=3    At ~2000 examples, more begins memorising rather than generalising.
alpha=2*r   Convention. alpha/r is an output SCALING factor, not extra capacity,
            which is why changing r does not force re-tuning the learning rate.
target_modules
            Attention AND MLP projections. Adapting attention only is the common
            shortcut and measurably underperforms.
fp16        The T4 is a Turing GPU with NO bf16 support. Using bf16 here fails.
sdpa        flash_attention_2 requires Ampere or newer; on a T4 it will not load.
paged_adamw_8bit
            8-bit optimizer states, paged to host RAM on a spike. Optimizer state
            is often the thing that OOMs, not the weights.

A NOTE ON WARMUP
----------------
transformers 5.x removed `warmup_ratio`; only `warmup_steps` survives. We
therefore compute the step count ourselves from dataset size, batch size,
gradient accumulation and epochs. Being explicit is no worse than the removed
convenience argument, and it makes the schedule inspectable.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.tokenization import PadCollator, build_training_example

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
WARMUP_RATIO = 0.03


def build_dataset(path, tokenizer, max_seq_length, limit=None):
    """Load the JSONL produced by data_prep and tokenize with loss masking."""
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if limit:
        records = records[:limit]

    examples = []
    for record in records:
        example = build_training_example(
            record["messages"], tokenizer, max_seq_length
        )
        if example is not None:  # data_prep already filtered; belt and braces
            examples.append(example)
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--output-dir", default="adapters/qwen-healthcare-lora")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--limit", type=int, default=None, help="Use N examples only (smoke test)."
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    dataset = build_dataset(
        args.train_file, tokenizer, args.max_seq_length, args.limit
    )

    # An optimizer step happens once per (batch_size * grad_accum) examples,
    # NOT once per batch — that is the whole point of gradient accumulation.
    examples_per_step = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(dataset) / examples_per_step)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = max(1, int(WARMUP_RATIO * total_steps))
    print(
        f"Training on {len(dataset)} examples | "
        f"{examples_per_step} examples/step | {total_steps} steps | "
        f"{warmup_steps} warmup steps"
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NF4 is information-theoretically better
                                    # than plain int4 for normally-distributed
                                    # neural network weights.
        bnb_4bit_use_double_quant=True,  # quantizes the quantization constants
        bnb_4bit_compute_dtype=torch.float16,  # fp16: T4 has no bf16
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        attn_implementation="sdpa",  # flash_attention_2 needs Ampere+
    )
    # Casts layernorms to fp32 and enables input gradients, both of which
    # 4-bit training needs to remain numerically stable.
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",   # attention
            "gate_proj", "up_proj", "down_proj",      # MLP
        ],
    )
    model = get_peft_model(model, lora_config)
    # Prints trainable vs total. Expect roughly 1% trainable — if it prints
    # 100%, LoRA did not attach and you are about to full-fine-tune by accident.
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,          # NOT bf16 — see module docstring
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none",   # no wandb prompt in Colab
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    trainer.train()

    # Saves ONLY the adapter (~40 MB), not the 3 GB base model. The adapter is
    # a diff and is meaningless without the base, which adapter_config.json
    # records by name.
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
