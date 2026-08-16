"""Smoke test: the environment can load the tokenizer we build every prompt with.

If this fails, nothing downstream can work, so it is worth its own test.
"""

from transformers import AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def test_tokenizer_loads_and_has_chat_template():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    # Every prompt in this project is rendered through the chat template.
    # A base (non-Instruct) model would have no template and silently
    # produce garbage prompts, so assert it exists.
    assert tok.chat_template is not None


def test_chat_template_produces_chatml_markers():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    # Qwen uses ChatML. The generation prompt must end with the assistant
    # header so the model knows it is its turn to speak.
    assert "<|im_start|>user" in text
    assert text.rstrip().endswith("<|im_start|>assistant")
