import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math, random, torch
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
# token budgets, not char budgets: chars per token swings ~2x between files, so a fixed
# character window still produced 132..4096-token samples and dragged the loss with it
FIM_NEED = MAXLEN - 5
# real /infill traffic fills a median of 2 tokens (p90 10, p99 128), so the hole is sampled
# log-uniform: most samples teach short inserts, the tail still teaches long boilerplate fills
FIM_HOLE_MIN, FIM_HOLE_MAX = 2, 256
ACCUM = 16
SHUFFLE = 256
SAVE_EVERY = 200
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
    total, count, correct, exact = h.sum() * 0, 0, 0, True
    def f(hh, yy):
        logits = (hh @ w.T).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
        loss = torch.nn.functional.cross_entropy(logits, yy, ignore_index=-100, reduction="sum")
        hit = (logits.argmax(-1) == yy) & (yy != -100)
        return loss, hit.sum(), (hit | (yy == -100)).all()
    for i in range(0, h.size(0), chunk):
        y = labels[i:i + chunk]
        n = (y != -100).sum().item()
        if n == 0:
            continue
        loss, hits, em = checkpoint(f, h[i:i + chunk], y, use_reentrant=False)
        total = total + loss
        correct += int(hits)
        exact = exact and bool(em)
        count += n
    return total / max(count, 1), correct, count, exact

def codeStream():
    # the dataset arrives repo by repo, so a small buffer shuffle breaks the long runs of similar files
    ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True, token=HF_TOKEN)
    buf = []
    for repo in ds:
        for f in repo["files"]:
            code = f.get("content")
            if not (code and (f.get("file_path") or "").endswith(EXT) and len(code) < 60000):
                continue
            ids = tok(code, add_special_tokens=False)["input_ids"]
            if len(ids) < FIM_NEED:
                continue
            buf.append(ids[:2 * FIM_NEED])
            if len(buf) >= SHUFFLE:
                yield buf.pop(random.randrange(len(buf)))
    random.shuffle(buf)
    yield from buf

PRE_ID, SUF_ID, MID_ID = (tok.convert_tokens_to_ids(t) for t in (FIM_PRE, FIM_SUF, FIM_MID))

def makeFim(ids):
    # the hole varies like real traffic but the total length stays exactly MAXLEN, so a loss
    # spike can only mean difficulty, never a longer window
    start = random.randint(0, len(ids) - FIM_NEED)
    w = ids[start:start + FIM_NEED]
    hole = int(2 ** random.uniform(math.log2(FIM_HOLE_MIN), math.log2(FIM_HOLE_MAX)))
    cut = random.randint(FIM_NEED // 3, FIM_NEED - hole - 1)
    pf, mid, sf = w[:cut], w[cut:cut + hole], w[cut + hole:]
    out = [tok.bos_token_id, PRE_ID] + pf + [SUF_ID] + sf + [MID_ID] + mid + [tok.eos_token_id]
    labels = [-100] * (4 + len(pf) + len(sf)) + mid + [tok.eos_token_id]
    return out, labels

def trainStep(ids, labels):
    ids = torch.tensor([ids]).cuda()
    labels = torch.tensor(labels[1:] + [-100]).cuda()
    h = backbone(input_ids=ids).last_hidden_state[0]
    loss, correct, tokens, exact = chunkLoss(h, headW, labels)
    (loss / ACCUM).backward()
    return loss.item(), correct, tokens, exact

if __name__ == "__main__":
    runId = os.environ.get("WANDB_RUN_ID")
    stepBase = int(os.environ.get("STEP_OFFSET", 0))
    wb.init(project="nemotron-fim-lora", id=runId, resume="allow" if runId else None)
    print("Starting training...")
    lossSum, correctSum, tokenSum, emSum = 0.0, 0, 0, 0
    for step, doc in enumerate(codeStream()):
        ids, labels = makeFim(doc)
        loss, correct, nTok, exact = trainStep(ids, labels)
        lossSum += loss
        correctSum += correct
        tokenSum += nTok
        emSum += int(exact)
        if (step + 1) % ACCUM == 0:
            gradNorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            acc = 100 * correctSum / max(tokenSum, 1)
            print(f"Step {stepBase + step + 1}, Loss {lossSum / ACCUM:.4f}, Acc {acc:.2f}%")
            wb.log({"loss": lossSum / ACCUM, "loss_last": loss, "lr": sched.get_last_lr()[0],
                    "grad_norm": gradNorm.item(), "seq_len": len(ids),
                    "token_acc": acc, "exact_match": 100 * emSum / ACCUM}, step=stepBase + step + 1)
            lossSum, correctSum, tokenSum, emSum = 0.0, 0, 0, 0
        if step % SAVE_EVERY == SAVE_EVERY - 1:
            peft.save_pretrained("out/lora")