"""Add tokenizer.ggml.fim_{pre,suf,mid}_token_id to an existing GGUF.

GGUF tensor offsets are relative to the tensor-data section, so growing the
metadata block only shifts where that section starts; the tensor payload can be
copied byte for byte.
"""
import struct, sys
from pathlib import Path

UINT32 = 4
ALIGN_KEY = "general.alignment"
DEFAULT_ALIGN = 32


def read_str(buf, pos):
    n = struct.unpack_from("<Q", buf, pos)[0]
    pos += 8
    return buf[pos:pos + n], pos + n


def read_value(buf, pos, vtype):
    if vtype == 8:                      # STRING: keep the length prefix
        start = pos
        _, pos = read_str(buf, pos)
        return buf[start:pos], pos
    if vtype == 9:                      # ARRAY
        itype = struct.unpack_from("<I", buf, pos)[0]
        count = struct.unpack_from("<Q", buf, pos + 4)[0]
        start = pos
        pos += 12
        for _ in range(count):
            _, pos = read_value(buf, pos, itype)
        return buf[start:pos], pos
    size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[vtype]
    return buf[pos:pos + size], pos + size


def main(src, dst, pre, suf, mid):
    buf = Path(src).read_bytes()
    assert buf[:4] == b"GGUF", "not a GGUF file"
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    print(f"version {version}  tensors {n_tensors}  kv {n_kv}")

    pos = 24
    kvs, align = [], DEFAULT_ALIGN
    for _ in range(n_kv):
        key, pos = read_str(buf, pos)
        vtype = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        raw, pos = read_value(buf, pos, vtype)
        kvs.append((key, vtype, raw))
        if key.decode() == ALIGN_KEY:
            align = struct.unpack_from("<i", raw, 8)[0]   # skip the string length prefix
    kv_end = pos
    print(f"metadata ends at {kv_end}, alignment {align}")

    infos_start = kv_end           # gguf-py writes tensor infos immediately after the metadata
    pos = infos_start
    for _ in range(n_tensors):
        _, pos = read_str(buf, pos)
        n_dims = struct.unpack_from("<I", buf, pos)[0]
        pos += 4 + 8 * n_dims + 4 + 8
    infos_end = pos
    data_start = (infos_end + align - 1) // align * align
    print(f"tensor infos {infos_start}-{infos_end}, data starts {data_start}")

    existing = {k.decode() for k, _, _ in kvs}
    add = []
    for name, tid in (("tokenizer.ggml.fim_pre_token_id", pre),
                      ("tokenizer.ggml.fim_suf_token_id", suf),
                      ("tokenizer.ggml.fim_mid_token_id", mid)):
        if name in existing:
            print(f"  {name} already present, skipping")
            continue
        add.append((name.encode(), UINT32, struct.pack("<I", tid)))
        print(f"  + {name} = {tid}")

    head = b"GGUF" + struct.pack("<IQQ", version, n_tensors, n_kv + len(add))
    for key, vtype, raw in kvs + add:
        head += struct.pack("<Q", len(key)) + key + struct.pack("<I", vtype) + raw

    out = bytearray(head)          # tensor infos follow the metadata unaligned
    out += buf[infos_start:infos_end]
    out += b"\x00" * (((len(out) + align - 1) // align * align) - len(out))
    out += buf[data_start:]

    shutil.copy2(src, dst + ".tmp") if False else None
    Path(dst).write_bytes(bytes(out))
    print(f"wrote {dst} ({len(out)/1e9:.2f} GB, was {len(buf)/1e9:.2f} GB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
