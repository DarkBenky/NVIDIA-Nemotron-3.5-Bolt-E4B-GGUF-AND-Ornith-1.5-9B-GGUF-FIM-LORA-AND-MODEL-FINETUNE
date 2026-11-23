import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random, torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from env import MODEL_NAME, MODEL_PATH, TRACE_PATH, UNUSED_TOKENS, HF_TOKEN, MODEL_PATH_SAVE_TENSOR
import wandb as wb

FIM_PRE, FIM_SUF, FIM_MID = UNUSED_TOKENS[:3]
BASE = MODEL_PATH_SAVE_TENSOR
FIM_LEN = 4096
MAXLEN = FIM_LEN
ACCUM = 16
EXT = (".c", ".h", ".go", ".py", ".vue", ".sql", ".js", ".ts", ".html", ".cpp", ".rs")

def loadBase(check=True):
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    if check:
        x = tok("def fibonacci(n):", return_tensors="pt").to("cuda")
        out = model.generate(**x, max_new_tokens=60, do_sample=False)
        print(tok.decode(out[0]))
    # text-only training: the official checkpoint also carries vision/audio towers we never call
    for name in ("vision_tower", "audio_tower", "embed_vision", "embed_audio"):
        if hasattr(model.model, name):
            delattr(model.model, name)
    return tok, model

tok, model = loadBase()
model.config.get_text_config().use_cache = False

backbone = getattr(model.model, "language_model", model.model)
backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
headW = model.get_output_embeddings().weight
cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

# llama.cpp only supports LoRA on linear layers, so no embeddings or lm_head
cfg = LoraConfig(
    r=128, lora_alpha=128, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=r".*\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
)
peft = (PeftModel.from_pretrained(model, os.environ["RESUME_LORA"], is_trainable=True)
        if os.environ.get("RESUME_LORA") else get_peft_model(model, cfg))
peft.print_trainable_parameters()
peft.train()  # GradientCheckpointingLayer only checkpoints when self.training, and from_pretrained returns eval mode

params = [p for p in peft.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))

def chunkLoss(h, w, labels, chunk=512):
    # a chunk of C costs C*vocab*4 bytes three times over (logits, log_softmax, grad): 2048 -> ~6 GB spike
    # total starts as a graph-connected zero: an all-masked sample leaves it untouched, and a plain 0.0
    # would make backward() fail because a float has no grad
    total, count = h.sum() * 0, 0
    def f(hh, yy):
        logits = (hh @ w.T).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return torch.nn.functional.cross_entropy(logits, yy, ignore_index=-100, reduction="sum")
    for i in range(0, h.size(0), chunk):
        y = labels[i:i + chunk]
        n = (y != -100).sum().item()
        if n == 0:
            continue
        total = total + checkpoint(f, h[i:i + chunk], y, use_reentrant=False)
        count += n
    return total / max(count, 1)

def codeStream():
    ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True, token=HF_TOKEN)
    for repo in ds:
        for f in repo["files"]:
            code = f.get("content")
            if code and (f.get("file_path") or "").endswith(EXT) and 300 < len(code) < 60000:
                yield code[:MAXLEN * 3]

def makeFim(code):
    # chars per token is ~3 here, so a window of FIM_CHARS chars yields roughly FIM_LEN tokens
    half = FIM_LEN * 3 // 2
    a, b = sorted(random.sample(range(len(code)), 2))
    b = min(b, a + half)
    pre = tok(f"{FIM_PRE}{code[max(0, a - half):a]}{FIM_SUF}{code[b:b + half]}{FIM_MID}", add_special_tokens=True)["input_ids"]
    mid = tok(code[a:b], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    ids = (pre + mid)[:MAXLEN]
    labels = ([-100] * len(pre) + mid)[:MAXLEN]
    return ids, labels

def trainStep(ids, labels):
    ids = torch.tensor([ids]).cuda()
    labels = torch.tensor(labels[1:] + [-100]).cuda()
    h = backbone(input_ids=ids).last_hidden_state[0]
    loss = chunkLoss(h, headW, labels)
    (loss / ACCUM).backward()
    return loss.item()

if __name__ == "__main__":
    wb.init(project="nemotron-fim-lora")
    print("Starting training...")
    for step, code in enumerate(codeStream()):
        ids, labels = makeFim(code)
        loss = trainStep(ids, labels)
        if (step + 1) % ACCUM == 0:
            gradNorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            print(f"Step {step+1}, Loss {loss:.4f}")
            wb.log({"loss": loss, "lr": sched.get_last_lr()[0],
                    "grad_norm": gradNorm.item(), "seq_len": len(ids)}, step=step + 1)
        if step % 1000 == 999:
            peft.save_pretrained("out/lora")