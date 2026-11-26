from transformers import AutoModelForCausalLM
import env
from train_lora_fim_common import train

# Gemma4 has no FIM tokens in its vocab, so three <unusedN> ids are repurposed as the FIM markers
if __name__ == "__main__":
    train({
        "base": env.MODEL_PATH_SAVE_TENSOR,
        "autoClass": AutoModelForCausalLM,
        "fimTokens": env.UNUSED_TOKENS[:3],
        "dropTowers": ("vision_tower", "audio_tower", "embed_vision", "embed_audio"),
        "minFreeGiB": 14,
        "wandbProject": "nemotron-fim-lora",
        "saveDir": "out/lora",
    })
