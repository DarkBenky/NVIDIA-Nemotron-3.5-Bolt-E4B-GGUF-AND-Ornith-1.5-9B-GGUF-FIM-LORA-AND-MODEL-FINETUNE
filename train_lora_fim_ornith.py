import os
# BF16 shards are ~18G and the root disk has ~32G free; the 2T drive already holds the model cache
os.environ.setdefault("HF_HOME", "/media/user/2TB/huggingface_cache")

from transformers import AutoModelForImageTextToText
from train_lora_fim_common import train

# Ornith already ships real FIM tokens (248060/248061/248062), so llama.cpp picks them up by name
# and /infill needs no token injection. The ConditionalGeneration wrapper (not the CausalLM one) is
# required: only it nests the text stack under model.language_model, which is how the weights are named
if __name__ == "__main__":
    train({
        "base": "ornith-ai/Ornith-1.5-9B",
        "autoClass": AutoModelForImageTextToText,
        "fimTokens": ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"),
        "dropTowers": ("visual",),
        "offloadEmbed": True,   # embed_tokens -> 3060 frees 1.89 GiB here
        "minFreeGiB": 20,
        "wandbProject": "ornith-fim-lora",
        "saveDir": "out/lora-ornith",
    })
