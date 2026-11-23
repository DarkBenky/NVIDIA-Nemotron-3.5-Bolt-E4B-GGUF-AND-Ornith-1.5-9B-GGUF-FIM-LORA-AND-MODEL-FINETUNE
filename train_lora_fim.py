import os, random, torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer
from env import MODEL_NAME, MODEL_PATH, TRACE_PATH

GGUF = MODEL_PATH
BASE = "out/base"
MAXLEN = 131072
ACCUM = 16
EXT = (".c", ".h", ".go", ".py", ".vue", ".sql")

def loadBase(check=True):
    if not os.path.isdir(BASE):
        d, f = os.path.split(GGUF)
        m = AutoModelForCausalLM.from_pretrained(d, gguf_file=f, dtype=torch.bfloat16)
        m.save_pretrained(BASE, safe_serialization=True)
        AutoTokenizer.from_pretrained(d, gguf_file=f).save_pretrained(BASE)
        del m
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda()
    if check:
        x = tok("def fibonacci(n):", return_tensors="pt").to("cuda")
        out = model.generate(**x, max_new_tokens=60, do_sample=False)
        print(tok.decode(out[0]))
    return tok, model

tok, model = loadBase()
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

backbone = getattr(model.model, "language_model", model.model)
headW = model.get_output_embeddings().weight
cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

# llama.cpp only supports LoRA on linear layers, so no embeddings or lm_head
cfg = LoraConfig(
    r=128, lora_alpha=128, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=r".*\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
)
peft = get_peft_model(model, cfg)
peft.print_trainable_parameters()

params = [p for p in peft.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50))

def chunkLoss(h, w, labels, chunk=2048):
    total, count = 0.0, 0
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
    ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True)
    for repo in ds:
        for f in repo["files"]:
            code = f.get("content")
            if code and (f.get("path") or "").endswith(EXT) and 300 < len(code) < 60000:
                yield code[:MAXLEN * 3]

def makeFim(code):
    a, b = sorted(random.sample(range(len(code)), 2))
    pre = tok(f"<unused0>{code[:a]}<unused1>{code[b:]}<unused2>", add_special_tokens=True)["input_ids"]
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
    print("Starting training...")
    for step, code in enumerate(codeStream()):
        ids, labels = makeFim(code)
        loss = trainStep(ids, labels)
        if (step + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            print(f"Step {step+1}, Loss {loss:.4f}")
        if step % 1000 == 999:
            peft.save_pretrained("out/lora")