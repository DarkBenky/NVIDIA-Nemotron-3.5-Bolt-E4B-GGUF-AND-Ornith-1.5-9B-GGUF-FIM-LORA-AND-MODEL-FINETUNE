#!/bin/bash
# Point both 8030 profiles at a fimfix GGUF + LoRA, with native FIM.
# usage: deploy_model.sh <gguf> <lora>
set -eu

GGUF=${1:?usage: deploy_model.sh <gguf> <lora>}
LORA=${2:?usage: deploy_model.sh <gguf> <lora>}
STAMP=$(date +%Y%m%d)

for unit in llama-server-zeta llama-server-zeta-3060; do
    f="$HOME/.config/systemd/user/$unit.service"
    cp -a "$f" "$f.bak-$STAMP"
    python3 - "$f" "$GGUF" "$LORA" <<'PY'
import re, sys
path, gguf, lora = sys.argv[1:4]
src = open(path).read()
# replace the -m line (model path) and inject --lora right after it
src = re.sub(r'(-m )\S+', r'\1' + gguf, src, count=1)
if '--lora ' not in src:
    src = re.sub(r'(-m \S+ \\\n)', r'\1    --lora ' + lora + ' \\\n', src, count=1)
# 65536 is the trained ceiling; the infill prefix is capped at 3/4 * batch-size
src = re.sub(r'--ctx-size \d+', '--ctx-size 65536', src)
src = re.sub(r'--batch-size \d+', '--batch-size 65520', src)
open(path, 'w').write(src)
PY
    echo "patched $unit (backup $f.bak-$STAMP)"
done

systemctl --user daemon-reload
echo
echo "=== diff ==="
for unit in llama-server-zeta llama-server-zeta-3060; do
    diff -u "$HOME/.config/systemd/user/$unit.service.bak-$STAMP" \
            "$HOME/.config/systemd/user/$unit.service" || true
done
