import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math, random, torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer
from env import (HF_TOKEN, ACCUM as ACCUM_DEFAULT, CLIP as CLIP_DEFAULT,
                 LORA_ALPHA as LORA_ALPHA_DEFAULT, LORA_R as LORA_R_DEFAULT,
                 LR_PEAK as LR_PEAK_DEFAULT, LR_TAIL as LR_TAIL_DEFAULT,
                 OPT_8BIT as OPT_8BIT_DEFAULT)
import wandb as wb

FIM_LEN = 4096
MAXLEN = FIM_LEN
# token budgets, not char budgets: chars per token swings ~2x between files
FIM_NEED = MAXLEN - 5
FIM_MIN = 256
# real /infill fills a median of 2 tokens, but FIM training standard (SantaCoder/BigCode) splits
# documents with uniformly random cut points, giving ~1200-token spans. Mix both: short inserts
# for autocomplete, random spans so suffix attention is actually exercised
FIM_HOLE_SHORT = 0.50    # share landing in 2-16 tokens
FIM_HOLE_MED = 0.25      # share landing in 17-256 tokens; the rest is a standard random span
ACCUM = int(os.environ.get("ACCUM", ACCUM_DEFAULT))
SHUFFLE = 256
SAVE_EVERY = 200
VALID_DIR = "/home/user/Desktop/fimTester/testcorpus"
VAL_SAMPLES = 160
VAL_EVERY = ACCUM * 25
EXT = (".c", ".h", ".go", ".py", ".vue", ".sql", ".js", ".ts", ".html", ".cpp", ".rs")

LR_PEAK = float(os.environ.get("LR_PEAK", LR_PEAK_DEFAULT))
LR_TAIL = int(os.environ.get("LR_TAIL", LR_TAIL_DEFAULT))
CLIP = float(os.environ.get("CLIP", CLIP_DEFAULT))
LORA_R = int(os.environ.get("LORA_R", LORA_R_DEFAULT))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", LORA_ALPHA_DEFAULT))
OPT_8BIT = int(os.environ.get("OPT_8BIT", OPT_8BIT_DEFAULT))

tok = model = backbone = headW = cap = peft = None
params, opt, sched = [], None, None
PRE_ID = SUF_ID = MID_ID = None
HEAD_IDS = []
TRAINABLE_PARAMS = TRAINABLE_BYTES = 0

def loadBase(profile):
    # .cuda() picks cuda:0, which without CUDA_DEVICE_ORDER is the fastest card (the 3090), not
    # nvidia-smi's PCI-ordered device 0. Fail before a long download instead of on a mid-load OOM
    freeGiB = torch.cuda.mem_get_info()[0] / 2 ** 30
    if freeGiB < profile["minFreeGiB"]:
        raise SystemExit(f"{torch.cuda.get_device_name(0)}: need ~{profile['minFreeGiB']}G free, "
                         f"have {freeGiB:.1f}G - stop llama-server-zeta and the router first")
    t = AutoTokenizer.from_pretrained(profile["base"], token=HF_TOKEN)
    m = profile["autoClass"].from_pretrained(profile["base"], dtype=torch.bfloat16, token=HF_TOKEN).cuda()
    try:
        x = t("def fibonacci(n):", return_tensors="pt").to("cuda")
        print(t.decode(m.generate(**x, max_new_tokens=60, do_sample=False)[0]))
    except Exception as e:
        print(f"generation check skipped: {type(e).__name__}: {str(e)[:120]}")
    # text-only: skip the vision/audio towers the checkpoint also carries
    for name in profile["dropTowers"]:
        if hasattr(m.model, name):
            delattr(m.model, name)
    return t, m

def setup(profile):
    global tok, model, backbone, headW, cap, peft, params, opt, sched, PRE_ID, SUF_ID, MID_ID, HEAD_IDS
    global TRAINABLE_PARAMS, TRAINABLE_BYTES
    tok, model = loadBase(profile)
    model.config.get_text_config().use_cache = False

    backbone = getattr(model.model, "language_model", model.model)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # multimodal wrappers return None from get_output_embeddings, but both carry a real lm_head
    head = model.get_output_embeddings()
    headW = (head if head is not None else model.lm_head).weight
    cap = getattr(model.config.get_text_config(), "final_logit_softcapping", None)

    PRE_ID, SUF_ID, MID_ID = (tok.convert_tokens_to_ids(x) for x in profile["fimTokens"])
    # Qwen3.5 has no BOS token, Gemma4 does, so the label mask length follows the tokenizer
    HEAD_IDS = [i for i in (tok.bos_token_id, PRE_ID) if i is not None]

    # llama.cpp only supports LoRA on linear layers, so no embeddings or lm_head
    cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=r".*\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    )
    peft = (PeftModel.from_pretrained(model, os.environ["RESUME_LORA"], is_trainable=True)
            if os.environ.get("RESUME_LORA") else get_peft_model(model, cfg))
    peft.print_trainable_parameters()
    peft.train()  # GradientCheckpointingLayer only checkpoints when self.training, and from_pretrained returns eval mode

    # embed_tokens is 1.89 GiB, frozen by PEFT, and read once per step as a table lookup, so parking
    # it on the second card costs one seq x 4096 bf16 copy of its output instead of the ~1 s/step that
    # moving 0.40 GiB layers over the x1 link would cost (their compute comes along, this has none)
    if profile.get("offloadEmbed") and torch.cuda.device_count() > 1:
        emb, compDev = backbone.embed_tokens, headW.device
        emb.to("cuda:1")
        emb.register_forward_pre_hook(lambda m, a: (a[0].to(m.weight.device),) + tuple(a[1:]))
        # checkpointing saves this tensor as layer 0's input, so the recompute re-reads it in place
        emb.register_forward_hook(lambda m, a, out: out.to(compDev))

    params = [p for p in peft.parameters() if p.requires_grad]
    # the adapter is exactly these tensors, so numel*itemsize is the size llama.cpp will load
    TRAINABLE_PARAMS = sum(p.numel() for p in params)
    TRAINABLE_BYTES = sum(p.numel() * p.element_size() for p in params)
    # the two FP32 Adam moments are 1.74 GiB for this adapter; 8-bit cuts them to 0.43 GiB
    if OPT_8BIT:
        import bitsandbytes as bnb
        opt = bnb.optim.Adam8bit(params, lr=LR_PEAK)
    else:
        opt = torch.optim.AdamW(params, lr=LR_PEAK, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lrScale)

def lrScale(s):
    if s < 50:
        return (s + 1) / 50
    if LR_TAIL <= 0:
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - 50) / LR_TAIL)))

# the fp32 logits are chunk x vocab x 4 bytes: 512 tokens needs 485 MiB for the chunk alone and
# 1.0 GiB peak, 128 tokens needs 121 MiB and 0.28 GiB. Measured 0.30 s either way, so chunk small
def chunkLoss(h, w, labels, chunk=128):
    # starts as a graph-connected zero so an all-masked sample still has a grad path
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

def loadValid():
    samples = []
    for lang in sorted(os.listdir(VALID_DIR)):
        d = os.path.join(VALID_DIR, lang)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            lines = open(os.path.join(d, name), errors="ignore").read().split("\n")
            for start, span in ((len(lines) // 4, 1), (len(lines) // 4, 3), (len(lines) // 2, 1),
                                (len(lines) // 2, 3), (3 * len(lines) // 4, 1), (3 * len(lines) // 4, 3)):
                md = tok("\n".join(lines[start:start + span]), add_special_tokens=False)["input_ids"]
                pf = tok("\n".join(lines[:start]), add_special_tokens=False)["input_ids"]
                sf = tok("\n".join(lines[start + span:]), add_special_tokens=False)["input_ids"]
                if not md or len(pf) < 8 or len(sf) < 4:
                    continue
                labels = [-100] * (len(HEAD_IDS) + 2 + len(pf) + len(sf)) + md + [tok.eos_token_id]
                samples.append((HEAD_IDS + pf + [SUF_ID] + sf + [MID_ID] + md + [tok.eos_token_id], labels))
    random.Random(0).shuffle(samples)
    return samples[:VAL_SAMPLES]

@torch.no_grad()
def validStep(valid):
    wasTraining = backbone.training
    backbone.eval()
    loss, count, hit, exact, eosP = 0.0, 0, 0, 0, 0.0
    for ids, labels in valid:
        h = backbone(input_ids=torch.tensor([ids]).cuda()).last_hidden_state[0]
        y = torch.tensor(labels[1:] + [-100]).cuda()
        ok = True
        for i in range(0, h.size(0), 512):
            yi = y[i:i + 512]
            if not (yi != -100).any():
                continue
            logits = (h[i:i + 512] @ headW.T).float()
            if cap:
                logits = torch.tanh(logits / cap) * cap
            loss += float(torch.nn.functional.cross_entropy(logits, yi, ignore_index=-100, reduction="sum"))
            pred = logits.argmax(-1)
            hit += int(((pred == yi) & (yi != -100)).sum())
            count += int((yi != -100).sum())
            ok = ok and bool(((pred == yi) | (yi == -100)).all())
            j = h.size(0) - 2   # -2 breaks when the last chunk is size 1, so index the absolute position instead
            if i <= j < i + logits.size(0):
                eosP += float(torch.softmax(logits[j - i], -1)[tok.eos_token_id])
        exact += int(ok)
    backbone.train(wasTraining)
    n = max(len(valid), 1)
    return {"val_loss": loss / max(count, 1), "val_acc": 100 * hit / max(count, 1),
            "val_em": 100 * exact / n, "val_eos_p": eosP / n}

def codeStream():
    # files arrive repo by repo, so shuffle within a buffer to break up similar runs
    ds = load_dataset("HuggingFaceCode/stack-v3-train", split="train", streaming=True, token=HF_TOKEN)
    buf = []
    for repo in ds:
        for f in repo["files"]:
            code = f.get("content")
            if not (code and (f.get("file_path") or "").endswith(EXT) and len(code) < 60000):
                continue
            ids = tok(code, add_special_tokens=False)["input_ids"]
            if len(ids) < FIM_MIN:
                continue
            buf.append(ids[:2 * FIM_NEED])
            if len(buf) >= SHUFFLE:
                yield buf.pop(random.randrange(len(buf)))
    random.shuffle(buf)
    yield from buf

def makeFim(ids):
    wlen = min(len(ids), int(2 ** random.uniform(math.log2(FIM_MIN), math.log2(FIM_NEED))))
    start = random.randint(0, len(ids) - wlen)
    w = ids[start:start + wlen]
    u = random.random()
    if u < FIM_HOLE_SHORT:
        hole = int(2 ** random.uniform(1, 4))
    elif u < FIM_HOLE_SHORT + FIM_HOLE_MED:
        hole = int(2 ** random.uniform(math.log2(17), 8))
    else:
        a, b = sorted(random.sample(range(wlen), 2))
        hole = b - a
    lo = min(64, wlen // 4)
    hole = min(hole, wlen - 2 * lo)
    cut = random.randint(lo, wlen - hole - lo)
    pf, mid, sf = w[:cut], w[cut:cut + hole], w[cut + hole:]
    out = HEAD_IDS + pf + [SUF_ID] + sf + [MID_ID] + mid + [tok.eos_token_id]
    labels = [-100] * (len(HEAD_IDS) + 2 + len(pf) + len(sf)) + mid + [tok.eos_token_id]
    return out, labels, hole

def trainStep(ids, labels):
    ids = torch.tensor([ids]).cuda()
    labels = torch.tensor(labels[1:] + [-100]).cuda()
    h = backbone(input_ids=ids).last_hidden_state[0]
    loss, correct, tokens, exact = chunkLoss(h, headW, labels)
    (loss / ACCUM).backward()
    return loss.item(), correct, tokens, exact

def train(profile):
    setup(profile)
    runId = os.environ.get("WANDB_RUN_ID")
    stepBase = int(os.environ.get("STEP_OFFSET", 0))
    wb.init(project=profile["wandbProject"], id=runId, resume="allow" if runId else None,
            config={"trainable_params": TRAINABLE_PARAMS, "adapter_mib": round(TRAINABLE_BYTES / 2 ** 20, 1)})
    saveDir = profile["saveDir"]
    print("Starting training...")
    valid = loadValid()
    print(f"validation set: {len(valid)} held-out holes")
    lossSum, correctSum, tokenSum, emSum = 0.0, 0, 0, 0
    holeBuf = []
    for step, doc in enumerate(codeStream()):
        ids, labels, hole = makeFim(doc)
        holeBuf.append(hole)
        loss, correct, nTok, exact = trainStep(ids, labels)
        lossSum += loss
        correctSum += correct
        tokenSum += nTok
        emSum += int(exact)
        if (step + 1) % ACCUM == 0:
            gradNorm = torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            acc = 100 * correctSum / max(tokenSum, 1)
            print(f"Step {stepBase + step + 1}, Loss {lossSum / ACCUM:.4f}, Acc {acc:.2f}%")
            hs = sorted(holeBuf)
            wb.log({"loss": lossSum / ACCUM, "loss_last": loss, "lr": sched.get_last_lr()[0],
                    "grad_norm": gradNorm.item(), "seq_len": len(ids),
                    "token_acc": acc, "exact_match": 100 * emSum / ACCUM,
                    "hole_mean": sum(hs) / len(hs),
                    "hole_med": hs[len(hs) // 2],
                    "hole_p90": hs[int(len(hs) * 0.9)],
                    "hole_le4_pct": 100 * sum(1 for h in hs if h <= 4) / len(hs),
                    "hole_ge64_pct": 100 * sum(1 for h in hs if h >= 64) / len(hs)}, step=stepBase + step + 1)
            if (step + 1) % VAL_EVERY == 0:
                v = validStep(valid)
                wb.log(v, step=stepBase + step + 1)
                print(f"  val loss {v['val_loss']:.4f}, acc {v['val_acc']:.2f}%, em {v['val_em']:.1f}%, p_eos {v['val_eos_p']:.3f}")
            if LR_TAIL > 0 and sched.last_epoch >= 50 + LR_TAIL:
                peft.save_pretrained(saveDir)
                print(f"LR tail finished at step {stepBase + step + 1}, stopping")
                break
            lossSum, correctSum, tokenSum, emSum = 0.0, 0, 0, 0
            holeBuf = []
        if step % SAVE_EVERY == SAVE_EVERY - 1:
            peft.save_pretrained(saveDir)

    print("Training finished.")
