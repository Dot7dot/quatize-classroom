"""咕咕夫教授的參考實作（Q4_0 量化）。

這個檔案是「教授的版本」：檢查器會拿它跟你寫的函式比對，
後面的章節如果你還沒通過任務，也會先借用它讓冒險能繼續。

想偷看解答的話…… 嗯，教授沒有鎖門 :)  但自己寫出來會更有成就感！

Q4_0 的規格（跟 llama.cpp 的 ggml 完全一致）：

* 每 32 個權重為一個 block
* 每個 block 存成 18 bytes：
    - 2 bytes ：scale ``d``（float16）
    - 16 bytes：32 個 4-bit 代碼（0~15），兩個擠一個 byte
* 編碼：找出「絕對值最大」的那個權重 ``m``（保留正負號），``d = m / -8``
        ``code = clip(trunc(x / d + 8.5), 0, 15)``
* 解碼：``x' = (code - 8) * d``
"""
from __future__ import annotations

import numpy as np

BLOCK = 32          # 每個 block 有幾個權重
BLOCK_BYTES = 18    # 每個 block 佔幾個 bytes（2 + 16）


# ---------------------------------------------------------------- 第 1 章
def model_size_mb(n_params, bits_per_weight):
    """模型大小（MB，1 MB = 1,000,000 bytes）。"""
    return n_params * bits_per_weight / 8 / 1_000_000


# ---------------------------------------------------------------- 第 2 章
def quantize(x, scale):
    """把浮點數量化成 4-bit 整數代碼（-8 ~ 7）。"""
    q = np.round(np.asarray(x, dtype=np.float32) / scale)
    return np.clip(q, -8, 7).astype(np.int8)


def dequantize(q, scale):
    """把整數代碼還原成浮點數。"""
    return np.asarray(q).astype(np.float32) * np.float32(scale)


# ---------------------------------------------------------------- 第 3 章
def absmax_scale(x):
    """讓 [-max|x|, +max|x|] 剛好落在 -7 ~ +7 的 scale。"""
    return float(np.max(np.abs(x))) / 7


def blockwise_roundtrip(x, block_size=32):
    """每個 block 用自己的 scale 量化再還原，回傳還原後的陣列。"""
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    for start in range(0, len(x), block_size):
        block = x[start:start + block_size]
        scale = absmax_scale(block)
        if scale == 0:
            out[start:start + block_size] = 0
            continue
        out[start:start + block_size] = dequantize(quantize(block, scale), scale)
    return out


def bits_per_weight(block_size, code_bits=4, scale_bits=16):
    """每個權重平均要花幾個 bit（含分攤的 scale）。"""
    return code_bits + scale_bits / block_size


# ---------------------------------------------------------------- 第 4 章
def q4_0_encode(block):
    """把 32 個 float32 權重編成 (d, codes)。

    d     ：float（已經先四捨五入成 float16 能表示的值）
    codes ：長度 32 的 uint8 陣列，每個值在 0~15
    """
    block = np.asarray(block, dtype=np.float32)
    assert block.shape == (BLOCK,), "一個 block 必須剛好 32 個權重"
    m = block[np.argmax(np.abs(block))]          # 絕對值最大者（保留正負號）
    d = np.float32(m) / np.float32(-8)
    inv = np.float32(0) if d == 0 else np.float32(1) / d
    codes = np.trunc(block * inv + np.float32(8.5))
    codes = np.clip(codes, 0, 15).astype(np.uint8)
    return float(np.float16(d)), codes


def q4_0_pack(d, codes):
    """把 (d, codes) 打包成 18 bytes。"""
    codes = np.asarray(codes, dtype=np.uint8)
    lo, hi = codes[:16], codes[16:]
    packed = lo | (hi << 4)                      # 低 4 bit 放前半、高 4 bit 放後半
    return np.float16(d).tobytes() + packed.astype(np.uint8).tobytes()


def q4_0_unpack(raw):
    """q4_0_pack 的反操作：18 bytes → (d, codes)。"""
    raw = bytes(raw)
    assert len(raw) == BLOCK_BYTES
    d = float(np.frombuffer(raw[:2], dtype=np.float16)[0])
    packed = np.frombuffer(raw[2:], dtype=np.uint8)
    codes = np.concatenate([packed & 0x0F, packed >> 4]).astype(np.uint8)
    return d, codes


def q4_0_decode(raw):
    """18 bytes → 32 個 float32 權重。"""
    d, codes = q4_0_unpack(raw)
    return (codes.astype(np.float32) - 8) * np.float32(d)


def q4_0_quantize(weights):
    """向量化版本：一次量化整個張量。

    weights：任意形狀的浮點陣列，但元素總數必須是 32 的倍數
    回傳   ：uint8 陣列，形狀 (block 數, 18)
    """
    blocks = np.asarray(weights, dtype=np.float32).reshape(-1, BLOCK)
    idx = np.abs(blocks).argmax(axis=1, keepdims=True)
    m = np.take_along_axis(blocks, idx, axis=1)              # (n, 1)
    d = m / np.float32(-8)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, np.float32(0), np.float32(1) / d).astype(np.float32)
    codes = np.trunc(blocks * inv + np.float32(8.5)).clip(0, 15).astype(np.uint8)
    packed = codes[:, :16] | (codes[:, 16:] << np.uint8(4))  # (n, 16)
    d_bytes = d.astype(np.float16).view(np.uint8)            # (n, 2)
    return np.concatenate([d_bytes, packed], axis=1)


def q4_0_dequantize(raw):
    """向量化版本：(block 數, 18) 的 uint8 → 一維 float32 權重。"""
    raw = np.ascontiguousarray(np.asarray(raw, dtype=np.uint8).reshape(-1, BLOCK_BYTES))
    d = raw[:, :2].copy().view(np.float16).astype(np.float32)     # (n, 1)
    packed = raw[:, 2:]
    codes = np.concatenate([packed & 0x0F, packed >> 4], axis=1).astype(np.float32)
    return ((codes - 8) * d).reshape(-1)
