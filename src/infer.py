"""Single-example inference — the live demo entry point.

Runs on CPU/MPS locally: the base model in fp16 is ~3 GB, which fits an 8 GB
Mac, so no GPU is needed to demo the result.

    python -m src.infer --task classify --text "burning when urinating, ..."
    python -m src.infer --task summarize --text "Doctor: What brings you in?..."

Add --no-adapter to run the same prompt through the untuned base model, which
is the fastest way to show the difference live.

THE DISCLAIMER LIVES HERE, NOT IN THE WEIGHTS
---------------------------------------------
It is deliberately printed by the wrapper rather than trained into the model.
Training it into the assistant turn would make it part of the learned output,
which would corrupt ROUGE against the references and waste adapter capacity on
a constant string. Safety text is an application-layer concern.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import build_classification_messages, build_summarization_messages
from src.tokenization import render_prompt

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

DISCLAIMER = (
    "\n---\nDecision-support demonstration only. Not a medical device and not "
    "medical advice. Outputs are unverified and must be reviewed by a qualified "
    "clinician."
)


def _resolve_device():
    """Pick the best local device and a dtype it actually supports.

    MPS (Apple's GPU backend) handles fp16 well. Plain CPU does not -- several
    ops fall back or run very slowly in fp16 -- so CPU gets fp32, at the cost
    of 6 GB instead of 3 GB.
    """
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _assert_no_meta_parameters(model):
    """Fail loudly if any weight is a placeholder rather than a real tensor.

    A `meta` parameter has a shape but no data. Loading an adapter into one
    silently succeeds and changes nothing, so without this check the demo would
    happily run the BASE model while claiming to run the tuned one. Turning a
    silent wrong answer into a crash is the whole point.
    """
    meta = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if meta:
        raise RuntimeError(
            f"{len(meta)} parameters are on the meta device (e.g. {meta[0]}). "
            "The model was offloaded and the adapter would be silently ignored."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["classify", "summarize"], required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--adapter", default="adapters/qwen-healthcare-lora")
    parser.add_argument("--labels", default="data/labels.json")
    parser.add_argument("--no-adapter", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # No 4-bit here: bitsandbytes has no Apple Silicon support, and at 1.5B
    # fp16 fits in 8 GB comfortably anyway. This is why infer.py does not reuse
    # evaluate.load_model, which quantizes.
    #
    # And deliberately NO device_map="auto". On an 8 GB Mac accelerate decides
    # it cannot fit the model, offloads layers to disk as `meta` (placeholder)
    # parameters, and PEFT then "loads" the adapter into those placeholders --
    # a documented no-op that discards the adapter. We load on CPU, attach the
    # adapter while every weight is real, and only then move to the device.
    device, dtype = _resolve_device()
    print(f"loading on {device} ({dtype})...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    if not args.no_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)

    _assert_no_meta_parameters(model)
    model.to(device)
    model.eval()

    if args.task == "classify":
        labels = json.loads(Path(args.labels).read_text())
        messages = build_classification_messages(args.text, labels)
        max_new_tokens = 16
    else:
        messages = build_summarization_messages(args.text)
        max_new_tokens = 256

    inputs = tokenizer(render_prompt(messages, tokenizer), return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(
        generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    print(answer)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
