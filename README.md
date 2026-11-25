# NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF-FIM-LORA-AND-MODEL-FINETUNE

FIM (fill-in-the-middle) LoRA for NVIDIA-Nemotron-3.5-Bolt-E4B, plus the local
model stack it is deployed into: a router on 8013, a Nemotron server on 8030, a
Qwen3.8-Flash-Next (Strata) server on 8080, a DeepSeek cloud fallback, and a
capture proxy that logs every request for future distillation.

## Local model topology

```
client (llama-vscode, Zed, fimTester, ...)
  |
  |  127.0.0.1:8013  llama-qwen-auto-router      ~/Desktop/fimTester/llama_model_router.py
  |
  +-- /v1/chat/completions ...... auto: Strata if loaded and free, else DeepSeek cloud
  +-- /small/v1/chat/...  ....... always the small local model, never cloud
  +-- /ds/v1/chat/completions ... always DeepSeek cloud, never local
  +-- /infill  /v1/completions .. always the small local model, via the capture proxy
      |
      +-- 127.0.0.1:8031  zeta-capture  (logging proxy)   ~/Desktop/zeta-proxy/zeta_proxy.py
          |
          +-- 127.0.0.1:8030  llama-server  (Nemotron-3.5-Bolt-E4B)

127.0.0.1:8080  strata-iq3s      Qwen3.8-Flash-Next IQ3_S, lazy, idle-unloads
127.0.0.1:8010  llama-embed      Nomic-Embed-Text-V2 (CPU only, RAG)
```

### Ports and units

| port | unit | model | notes |
|---|---|---|---|
| 8013 | `llama-qwen-auto-router` | — | routes; logs to `requests-YYYY-MM-DD.jsonl` |
| 8030 | `llama-server-zeta` | Nemotron UD-Q4_K_XL | small model, whole model on the RTX 3090 |
| 8030 | `llama-server-zeta-3060` | Nemotron UD-Q4_K_XL | same model on the RTX 3060, used while Strata is loaded |
| 8031 | `zeta-capture` | — | logging proxy 8031 -> 8030 |
| 8080 | `strata-iq3s` | Qwen3.8-Flash-Next IQ3_S | big model, `--lazy`, `idle_unload_s` frees VRAM |
| 8010 | `llama-embed` | Nomic-Embed-Text-V2 | embeddings for llama-vscode RAG, CPU |

The two 8030 profiles are mutually exclusive (`Conflicts=`) and are swapped by
the router, never by hand. Only one of them owns the port at a time.

### Routing policy

- **Default (fast)**: the small model sits on the 3090; the Strata server is up
  but its model is unloaded.
- **Big-mode switch**: when the Strata model is loaded and no other chat is
  running, a chat request stops the fast profile, starts the 3060 profile, and
  forwards to Strata. `/small/`, `/infill` and `/v1/completions` keep working
  the whole time because the small model only moved cards.
- **Cloud overflow** (needs `DEEPSEEK_API_KEY`): a chat request goes to DeepSeek
  when the local model cannot serve it immediately — another request is running
  (Strata serves one at a time), a profile switch is in progress, or the Strata
  model is idle-unloaded. The local model is warmed in the background either way.
- **Restore**: five minutes after Strata idle-sleeps, the fast 3090 profile comes
  back.

### Logging

Every POST that reaches the router is appended as one JSON line to
`ROUTER_LOG_DIR/requests-YYYY-MM-DD.jsonl` (default `/media/user/sde1/conversationsLLM`).
FIM traffic is additionally captured by the 8031 proxy into
`~/Desktop/fimTester/synteticData/data/fimEdit/captures.jsonl`, in the
`{input_prefix, input_suffix} -> response_content` shape used to build training
data. Set `ROUTER_LOG_SKIP_INFILL=1` to drop `/infill` from the router log.

## FIM tokens: why `/infill` used to return 501

`llama.cpp` detects FIM markers by name only (`<|fim_prefix|>`, `<fim_prefix>`,
`<PRE>`, `<|code_prefix|>`, ...) in `src/llama-vocab.cpp`. Nemotron reserves
`<unused0>` / `<unused1>` / `<unused2>` for FIM and matches none of them, so
`fim_pre_id` stays null and the server rejects `/infill` with HTTP 501. There is
no CLI flag to override it — the IDs must be present in the GGUF:

```
tokenizer.ggml.fim_pre_token_id = 6     (<unused0>)
tokenizer.ggml.fim_suf_token_id = 7     (<unused1>)
tokenizer.ggml.fim_mid_token_id = 8     (<unused2>)
```

Two ways to get them, both in this repo:

- `conversion/gemma.py` in `llama.cpp` — patched `Gemma4Model.set_vocab()` to
  emit the three keys for every gemma4 conversion. Prefer this for new work.
- `add_fim_tokens.py` — injects the keys into an existing GGUF without
  re-quantising. GGUF tensor offsets are relative to the data section, so the
  metadata block can grow and the tensor payload is copied byte for byte. All
  720 tensors verified identical (name, shape, type, data md5) after injection,
  and a no-op run is byte-identical to the input.

Until a deployed GGUF has these keys, the 8031 proxy shims `/infill` into a
chat turn ("Complete the missing code at `<CURSOR>`"), which costs accuracy
against native FIM.

## Training

`train_lora_fim.py`, LoRA `r=128` on attention and MLP projections
(279M of 7.74B params trainable), base = NVIDIA's official BF16 safetensors.

```bash
# fresh run
python -u train_lora_fim.py

# resume from a saved adapter, with a decaying LR tail that ends the run by itself
RESUME_LORA=out/lora-step75040 WANDB_RUN_ID=<id> STEP_OFFSET=75040 \
  LR_PEAK=3e-5 LR_TAIL=300 CLIP=2.5 python -u train_lora_fim.py
```

| env var | default | meaning |
|---|---|---|
| `RESUME_LORA` | — | adapter directory to resume from |
| `WANDB_RUN_ID` | — | wandb run to append to |
| `STEP_OFFSET` | `0` | printed step offset for resumed runs |
| `LR_PEAK` | `2e-4` | peak learning rate |
| `LR_TAIL` | `0` | cosine-decay length in optimizer steps; `0` = flat LR, no auto-stop |
| `CLIP` | `2.5` | gradient-norm clip |

### Sampling

Each sample is a token window cut from a real training file, tokenised once:

```
<bos> <unused0> prefix <unused1> suffix <unused2> hole <eos>
```

- Window length is log-uniform between 256 and 4095 tokens. A fixed 4096-token
  window never matched a real request; real `/infill` calls have a median
  window near 1000 tokens.
- Prefix is at least `min(64, wlen/4)` tokens. The earlier code used
  `FIM_NEED // 3`, which forced a 1363-token minimum and put every short-file
  sample out of distribution.
- Hole size is a mixture: 50% in 2-16 tokens, 25% in 17-256, 25% a uniform
  random span. Autocomplete is dominated by short inserts, so the short bucket
  carries the most mass.
- Loss is computed on the hole plus `<eos>` only.

### Held-out validation

Every 800 samples the trainer scores 160 holes built from
`~/Desktop/fimTester/testcorpus` (97 real files from this machine: C, Go,
Python, JS; holes at 1/4, 1/2 and 3/4 of each file, spans of 1 and 3 lines).
Logged to wandb:

| metric | meaning |
|---|---|
| `val_loss` | cross-entropy on the hole, teacher-forced |
| `val_acc` | token accuracy on the hole |
| `val_em` | exact reconstruction of the whole hole |
| `val_eos_p` | probability of `<eos>` right after the hole — the stop signal |

`val_eos_p` is the one that catches the failure mode training loss hides: a
model that keeps writing past the hole still scores well on token accuracy.
It is teacher-forced, so it measures "does it know where to stop", not "does it
stop in practice", and it is only meaningful next to `val_acc`.

### Results

Validation on the same 160 holes, pre-tail vs post-tail:

| checkpoint | val_loss | val_acc | val_em | val_eos_p |
|---|---|---|---|---|
| `lora-final-2148` (step 66240, pre-tail) | 2.0259 | 73.01% | 0.0% | 0.0029 |
| `lora-step69440` (after a 150-step tail) | 1.4584 | 73.72% | 0.0% | 0.1310 |

The tail cut `val_loss` by 28% and lifted the stop probability 45x while leaving
token accuracy flat — the adapter became better calibrated rather than more
accurate. `val_em` has never left 0%: it does not reproduce a whole hole exactly.

## Building and deploying a GGUF

```bash
# 1. LoRA adapter -> GGUF (must use the same base the adapter was trained on)
python ~/Desktop/llama.cpp/convert_lora_to_gguf.py out/lora \
  --base /media/user/2TB/huggingface_cache/hub/models--nvidia--NVIDIA-Nemotron-3.5-Bolt-E4B-BF16/snapshots/<sha> \
  --outfile /media/user/2TB/nemotron-fim-lora.gguf --outtype bf16

# 2. serve with native FIM
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  ~/Desktop/llama.cpp/build/bin/llama-server \
  -m /media/user/2TB/Nemotron-Bolt-E4B-fimfix-Q4_K_M.gguf \
  --lora /media/user/2TB/nemotron-fim-lora.gguf \
  -c 65536 -b 65520 -ub 512 --flash-attn on --jinja
```

`-b` matters more than `-c` for `/infill`: the server caps the prefix at
`3/4 * n_batch` and the suffix at `n_batch/4`, so `-b 16384` silently limits
every prompt to ~12k tokens no matter how large `-c` is. Both caps maxed out at
`-b 65536` overshoot the context by 2 tokens, hence 65520.

Measured on the RTX 3060 (Q4_K_M + LoRA, `-c 65536`): 4.5 GB VRAM, prefill
~1100-1270 t/s and decode ~35 t/s at 30k-50k token prompts, 2860 t/s and 88 t/s
at 600 tokens. The window is cheap because 35 of 42 layers use a 512-token
sliding window and 18 KV layers are shared.

## Repository layout

| file | purpose |
|---|---|
| `train_lora_fim.py` | LoRA FIM training, sampling, validation, resume |
| `env.py` | paths and the `<unused*>` token list (gitignored: contains an HF token) |
| `add_fim_tokens.py` | inject `tokenizer.ggml.fim_*_token_id` into an existing GGUF |

Adapters and logs are gitignored (`out/`, `wandb/`, `*.log`); checkpoints live
in `out/lora*` on disk only.

## Suspending local routing during training

Training needs ~23.5 of 24 GiB on the 3090, so the router must not start, stop
or load any local unit while a run is active.

```bash
cat > ~/.config/systemd/user/llama-qwen-auto-router.service.d/no-local.conf <<'EOF'
[Service]
Environment=ROUTER_NO_LOCAL=1
EOF
systemctl --user daemon-reload && systemctl --user restart llama-qwen-auto-router
```

With the flag set, `initial_setup()` and `monitor()` are skipped,
`ensure_shared_and_big()` raises instead of touching units, and `/ds/` (cloud)
keeps working. Restore with:

```bash
rm ~/.config/systemd/user/llama-qwen-auto-router.service.d/no-local.conf
systemctl --user daemon-reload && systemctl --user restart llama-qwen-auto-router
```

Without the flag, `initial_setup()` starts the 3090 profile and the idle
monitor's `restore_fast()` takes the card back, which OOM-kills training.
