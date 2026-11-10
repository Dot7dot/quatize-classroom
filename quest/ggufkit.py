"""GGUF 小幫手：看檔案、抄 metadata、比較兩個檔案、把權重裝回 PyTorch。

量化本身的邏輯不在這裡（那是你在 Notebook 裡親手寫的），
這裡只放「搬東西」的雜事。
"""
from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from . import toolbox, ui


def _gguf():
    import gguf
    return gguf


# ------------------------------------------------------------------ 看一個檔案
def summary(path) -> dict:
    """回傳一個 GGUF 檔的重點數字。"""
    gguf = _gguf()
    r = gguf.GGUFReader(str(path))
    types = Counter(t.tensor_type.name for t in r.tensors)
    n_weights = int(sum(int(t.n_elements) for t in r.tensors))
    data_bytes = int(sum(int(t.n_bytes) for t in r.tensors))
    ft = r.fields.get("general.file_type")
    return {
        "path": str(path),
        "size_mb": os.path.getsize(path) / 1e6,
        "n_tensors": len(r.tensors),
        "n_kv": len([k for k in r.fields if not k.startswith("GGUF.")]),
        "types": dict(types),
        "n_weights": n_weights,
        "bits_per_weight": data_bytes * 8 / max(1, n_weights),
        "file_type": int(ft.contents()) if ft is not None else None,
        "data_offset": int(r.data_offset),
    }


def show_summary(path, title: str = "") -> dict:
    s = summary(path)
    types = "、".join(f"{k} × {v}" for k, v in sorted(s["types"].items(), key=lambda kv: -kv[1]))
    ui.show_note(
        f"📦 {title or Path(path).name}",
        [f"檔案大小：<b>{s['size_mb']:.1f} MB</b>",
         f"張量：{s['n_tensors']} 個（{types}）",
         f"權重總數：{s['n_weights']:,} 個　→　平均每個權重 <b>{s['bits_per_weight']:.2f} bit</b>"],
        plain=f"{path}: {s['size_mb']:.1f} MB, {s['types']}, {s['bits_per_weight']:.2f} bpw",
    )
    return s


def short(value, limit: int = 60) -> str:
    """把很長的設定值縮短成一行，方便印出來。"""
    if isinstance(value, (list, tuple)):
        head = ", ".join(repr(v) for v in value[:4])
        text = f"[{head}, … 共 {len(value)} 個]" if len(value) > 4 else f"[{head}]"
    else:
        text = repr(value) if isinstance(value, str) else str(value)
    text = text.replace("\n", "\\n")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def map_data(path) -> dict:
    """給「GGUF 藏寶箱地圖」互動元件用的資料（全部來自你自己的檔案）。"""
    gguf = _gguf()
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(24)
    r = gguf.GGUFReader(str(path))
    hexs = lambda b: " ".join(f"{x:02X}" for x in b)  # noqa: E731
    kvs = []
    for key, field in r.fields.items():
        if key.startswith("GGUF."):
            continue
        kvs.append({"key": ui.esc(key), "type": field.types[-1].name.lower() + ("[]" if len(field.types) > 1 else ""),
                    "value": ui.esc(short(field.contents()))})
    wanted = ["general.architecture", "general.name", "llama.block_count", "llama.embedding_length",
              "llama.context_length", "llama.attention.head_count", "general.file_type",
              "tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.chat_template"]
    picked = [k for name in wanted for k in kvs if k["key"] == name]
    tensors = [{"name": ui.esc(t.name), "shape": " × ".join(str(int(d)) for d in t.shape),
                "type": t.tensor_type.name, "nbytes": int(t.n_bytes)} for t in r.tensors[:7]]
    n_kv = int.from_bytes(head[16:24], "little")
    return {
        "file": path.name, "size": os.path.getsize(path),
        "magic_hex": hexs(head[0:4]), "version_hex": hexs(head[4:8]), "nt_hex": hexs(head[8:16]),
        "nkv_hex": hexs(head[16:24]), "version": int.from_bytes(head[4:8], "little"),
        "n_tensors": len(r.tensors), "n_kv": n_kv, "kvs": picked or kvs[:10], "tensors": tensors,
        "data_offset": int(r.data_offset), "alignment": int(r.alignment),
    }


# ------------------------------------------------------------------ 做一個新的 GGUF
def should_quantize(tensor) -> bool:
    """這個張量要不要量化？（跟 llama-quantize 的基本規則一樣）

    * 二維以上的「weight」才量化 —— 它們佔了 99.9% 的空間
    * 一維的 norm 權重很小又很敏感，保持原樣
    """
    return len(tensor.shape) >= 2 and tensor.name.endswith("weight") and "_norm." not in tensor.name


def copy_metadata(reader, writer, file_type: int | None = None) -> int:
    """把 reader 裡所有的設定值抄到 writer。file_type 有給的話就順便改掉。回傳抄了幾筆。"""
    gguf = _gguf()
    n = 0
    for field in reader.fields.values():
        # GGUF.* 是讀檔時自動產生的；architecture 在建立 writer 時就寫過了
        if field.name.startswith("GGUF.") or field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        value = field.contents()
        if field.name == gguf.Keys.General.FILE_TYPE and file_type is not None:
            value = int(file_type)
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, value, vtype, sub_type=sub)
        n += 1
    return n


def to_float32(tensor) -> np.ndarray:
    """把 GGUF 裡的一個張量（不管它是 F16、F32 還是量化過的）解回 float32。"""
    gguf = _gguf()
    return np.asarray(gguf.quants.dequantize(np.asarray(tensor.data), tensor.tensor_type), dtype=np.float32)


# ------------------------------------------------------------------ 比較兩個 GGUF
def compare(mine, official) -> dict:
    """逐一比較兩個 GGUF 檔的張量。回傳統計數字，並顯示一張表。"""
    gguf = _gguf()
    a, b = gguf.GGUFReader(str(mine)), gguf.GGUFReader(str(official))
    bt = {t.name: t for t in b.tensors}
    same_type = diff_type = 0
    bytes_same = bytes_total = 0
    identical_tensors = 0
    diffs = []
    for t in a.tensors:
        o = bt.get(t.name)
        if o is None:
            continue
        if t.tensor_type != o.tensor_type:
            diff_type += 1
            diffs.append((t.name, t.tensor_type.name, o.tensor_type.name, int(t.n_bytes), int(o.n_bytes)))
            continue
        same_type += 1
        x, y = np.asarray(t.data).reshape(-1).view(np.uint8), np.asarray(o.data).reshape(-1).view(np.uint8)
        eq = int((x == y).sum())
        bytes_same += eq
        bytes_total += x.size
        identical_tensors += int(eq == x.size)
    rows = [
        ["檔案大小", f"{os.path.getsize(mine) / 1e6:.1f} MB", f"{os.path.getsize(official) / 1e6:.1f} MB"],
        ["張量數量", str(len(a.tensors)), str(len(b.tensors))],
        ["型別分布", _types(a), _types(b)],
    ]
    pct = 100.0 * bytes_same / max(1, bytes_total)
    note = (f"在 <b>{same_type}</b> 個型別相同的張量裡，有 <b>{identical_tensors}</b> 個連一個 byte 都不差；"
            f"全部加起來 <b>{pct:.3f}%</b> 的 byte 一模一樣。")
    if diffs:
        note += f"<br>另外有 <b>{diff_type}</b> 個張量，官方工具選了不一樣的型別（見下一張表）。"
    ui.show_table("🔍 你的手工版 vs 官方版", ["", "你的手工版", "官方 llama-quantize"], rows, note=note)
    if diffs:
        ui.show_table(
            "🧐 型別不一樣的張量", ["張量", "你的", "官方", "你的大小", "官方大小"],
            [[ui.esc(n), ta, f'<span style="color:{ui.C["gold"]}">{tb}</span>', f"{na / 1e6:.2f} MB", f"{nb / 1e6:.2f} MB"]
             for n, ta, tb, na, nb in diffs[:8]],
            note="官方工具不會傻傻地把每個張量都變成 Q4_0 —— 特別重要的張量會被「升級」成比較精細的型別。"
                 "這就是為什麼檔案型別叫 <b>MOSTLY</b>_Q4_0（大部分是 Q4_0）。")
    return {"same_type": same_type, "diff_type": diff_type, "identical_tensors": identical_tensors,
            "byte_match_pct": pct, "diffs": diffs}


def compare_with_reference(f16_path, quantizer) -> float:
    """沒有 llama-quantize 時的備案：拿官方 gguf 套件裡的 Q4_0 參考實作，跟你的量化器逐 byte 比對。"""
    gguf = _gguf()
    r = gguf.GGUFReader(str(f16_path))
    same = total = n = 0
    for t in r.tensors:
        if not should_quantize(t):
            continue
        w = np.asarray(t.data, dtype=np.float32)
        mine = np.asarray(quantizer(w)).reshape(-1)
        ref = np.asarray(gguf.quants.quantize(w, gguf.GGMLQuantizationType.Q4_0)).reshape(-1)
        if mine.shape != ref.shape:
            continue
        same += int((mine == ref).sum())
        total += ref.size
        n += 1
    pct = 100.0 * same / max(1, total)
    ui.show_note("🔍 你的量化器 vs 官方 gguf 套件的參考實作",
                 [f"比對了 {n} 個張量、{total:,} 個 byte：<b>{pct:.3f}%</b> 一模一樣。"],
                 plain=f"match {pct:.3f}% over {n} tensors")
    return pct


def _types(reader) -> str:
    c = Counter(t.tensor_type.name for t in reader.tensors)
    return "　".join(f"{k}×{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


# ------------------------------------------------------------------ 裝回 PyTorch 做檢定
def load_into_torch(model, gguf_path) -> int:
    """把 GGUF 檔裡的權重（解量化之後）裝進 Hugging Face 的 PyTorch 模型。回傳換掉了幾個張量。

    llama.cpp 執行量化模型時，數學上做的就是同一件事：用 (代碼 − 8) × d 還原出權重再去算。
    """
    import torch
    gguf = _gguf()

    r = gguf.GGUFReader(str(gguf_path))
    cfg = model.config
    n_head, n_kv = cfg.num_attention_heads, getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, cfg.num_hidden_layers)
    by_name = {t.name: t for t in r.tensors}
    state = model.state_dict()
    n = 0
    with torch.no_grad():
        for hf_name, param in state.items():
            key = hf_name if hf_name != "lm_head.weight" else "model.embed_tokens.weight"   # 輸出層共用詞嵌入
            gg = name_map.get_name(key, try_suffixes=(".weight", ".bias"))
            if gg is None or gg not in by_name:
                continue
            w = to_float32(by_name[gg]).reshape(tuple(param.shape))
            if key.endswith("q_proj.weight"):
                w = toolbox.unpermute_qk(w, n_head)
            elif key.endswith("k_proj.weight"):
                w = toolbox.unpermute_qk(w, n_kv)
            param.copy_(torch.tensor(w, dtype=param.dtype))
            n += 1
    return n


EXAM_TEXT = (
    "The llama is a domesticated animal from South America. For thousands of years, people living in the "
    "Andes mountains have used llamas to carry goods along steep and narrow paths. A llama can carry about "
    "a quarter of its own body weight, and it can walk for many hours without getting tired. Llamas are "
    "social animals and prefer to live in a herd. They are also very curious, and they will often walk up "
    "to people to take a closer look. When a llama is annoyed, it may spit, but it usually only does this "
    "to other llamas. Their wool is soft and warm, so it is used to make clothes, blankets and ropes. "
    "Today, llamas can be found on farms all over the world. Some farmers keep them to guard sheep, "
    "because a llama will chase away small predators. Children love to visit them, and many people say "
    "that spending time with a llama makes them feel calm and happy."
)


def perplexity(model, tokenizer, text: str = EXAM_TEXT) -> float:
    """困惑度（perplexity）：模型讀一段文章時，平均每個字「有多意外」。越低代表越懂這段文字。"""
    import torch
    ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        loss = model(ids, labels=ids).loss
    return float(math.exp(loss.item()))


def torch_generate(model, tokenizer, question: str, max_tokens: int = 80) -> str:
    """用 PyTorch 讓模型回答一個問題（固定挑機率最高的字，所以每次結果一樣）。"""
    import torch
    msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": question}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(ids, torch.Tensor):       # 新版 transformers 回傳的是 dict
        ids = ids["input_ids"]
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
