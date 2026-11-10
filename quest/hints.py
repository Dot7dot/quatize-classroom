"""提示與解答。

quest.hint("任務編號") 會依序顯示這裡的提示；提示用完之後才會顯示解答（經驗值減半）。
tests/ 裡的自動測試也會拿這裡的解答去跑整本 Notebook，確保教材本身沒有壞掉。
"""

HINTS = {
    "1-1": [
        "總共有 <code>n_params × bits_per_weight</code> 個 bit。",
        "8 個 bit 是 1 個 byte；1,000,000 個 byte 是 1 MB。",
    ],
    "1-2": [
        "把滑桿拉到讓「模型大小」剛好低於 100 MB 的最大整數。",
        "也可以用算的：100,000,000 bytes × 8 ÷ 134,515,008 ≈ 5.95，所以最多 5 bit。",
    ],
    "2-1": [
        "量化三步驟：<b>除以 scale → 四捨五入 → 夾進範圍</b>。",
        "四捨五入用 <code>np.round(...)</code>，夾住用 <code>np.clip(q, -8, 7)</code>。",
        "還原只要一步：<code>q * scale</code>。",
    ],
    "2-2": [
        "「最大可能誤差」就是半格：scale ÷ 2。",
        "5 bit 的最大誤差是 0.0333，6 bit 是 0.0161。",
    ],
    "3-1": [
        "<code>absmax_scale</code>：先取絕對值 <code>np.abs(block)</code>，找最大 <code>np.max(...)</code>，再除以 7。",
        "迴圈裡的三個空格分別是：<code>absmax_scale(block)</code>、<code>quantize(block, scale)</code>、<code>dequantize(q, scale)</code>。",
    ],
    "3-2": [
        "每塊要多存一個 16-bit 的 scale，由這一塊裡的所有權重平均分攤。",
        "公式：4 + 16 ÷ 每塊的權重數。每 4 個一塊 → 4 + 16 ÷ 4。",
    ],
    "4-1": [
        "找「絕對值最大、但保留正負號」的元素：<code>block[np.argmax(np.abs(block))]</code>。",
        "<code>d = m / -8</code>。注意是<b>負</b> 8。",
        "代碼：<code>np.trunc(block / d + 8.5)</code>，再用 <code>np.clip(..., 0, 15)</code> 夾住。",
    ],
    "4-2": [
        "d 佔 2 bytes：<code>np.float16(d).tobytes()</code>。",
        "前 16 個代碼是 <code>codes[:16]</code>，後 16 個是 <code>codes[16:]</code>。",
        "擠在一起：<code>lo | (hi << 4)</code>。<code>&lt;&lt; 4</code> 的意思是往左移 4 個 bit，也就是乘以 16。",
    ],
    "4-3": [
        "取出低 4 bit：<code>packed & 0x0F</code>；取出高 4 bit：<code>packed >> 4</code>。",
        "低 4 bit 是前 16 個、高 4 bit 是後 16 個，用 <code>np.concatenate([lo, hi])</code> 接起來。",
        "還原公式：<code>(codes - 8) * d</code>。記得先把 codes 轉成浮點數再減 8。",
    ],
    "4-4": [
        "公式永遠是：代碼 = 去掉小數( x ÷ d + 8.5 )，超過 15 就當 15。",
        "d 是負數的時候，數線上的代碼會左右顛倒，不要被騙了。",
    ],
    "4-5": [
        "每一步都跟單一 block 的版本一樣，只是多了 <code>axis=1</code>（對每一列各做一次）。",
        "兩個空格：<code>m / np.float32(-8)</code> 以及 <code>np.trunc(blocks * inv + np.float32(8.5))</code>。",
    ],
    "5-3": [
        "<code>struct.unpack</code> 的格式字串：<code>&lt;</code> 表示 little-endian，<code>4s</code> 是 4 個字元，<code>I</code> 是 32-bit 整數，<code>Q</code> 是 64-bit 整數。",
        "檔頭依序是：魔法數字（4 bytes）、版本（32-bit）、張量數量（64-bit）、設定數量（64-bit）。",
    ],
    "5-4": [
        "十六進位 47 是 G。那 55 和 46 呢？",
    ],
    "8-3": [
        "每一題都能在前面的章節找到答案。答錯沒關係，可以再選。",
    ],
}

# 程式題的解答（只有函式本體；檢查用的那一行留在 Notebook 裡）
SOLUTIONS = {
    "1-1": '''def model_size_mb(n_params, bits_per_weight):
    total_bits = n_params * bits_per_weight
    total_bytes = total_bits / 8
    return total_bytes / 1_000_000
''',
    "2-1": '''def quantize(x, scale):
    q = np.round(x / scale)          # 除以 scale，四捨五入到最近的整數
    q = np.clip(q, -8, 7)            # 4 bit 只能表示 -8 ~ 7
    return q.astype(np.int8)

def dequantize(q, scale):
    return q * scale
''',
    "3-1": '''def absmax_scale(block):
    return np.max(np.abs(block)) / 7

def blockwise_roundtrip(x, block_size):
    out = np.empty_like(x)
    for start in range(0, len(x), block_size):
        block = x[start:start + block_size]
        scale = absmax_scale(block)
        q = quantize(block, scale)
        out[start:start + block_size] = dequantize(q, scale)
    return out
''',
    "4-1": '''def q4_0_encode(block):
    m = block[np.argmax(np.abs(block))]      # 絕對值最大的權重（保留正負號）
    d = m / -8                               # scale
    codes = np.trunc(block / d + 8.5)        # 每個權重落在第幾格
    codes = np.clip(codes, 0, 15)            # 夾進 0 ~ 15
    return d, codes.astype(np.uint8)
''',
    "4-2": '''def q4_0_pack(d, codes):
    d_bytes = np.float16(d).tobytes()        # 2 bytes
    lo = codes[:16]                          # 前 16 個 → 低 4 bit
    hi = codes[16:]                          # 後 16 個 → 高 4 bit
    packed = lo | (hi << 4)                  # 16 bytes
    return d_bytes + packed.astype(np.uint8).tobytes()
''',
    "4-3": '''def q4_0_decode(raw):
    d = np.frombuffer(raw[:2], dtype=np.float16)[0]
    packed = np.frombuffer(raw[2:], dtype=np.uint8)
    lo = packed & 0x0F                       # 前 16 個代碼
    hi = packed >> 4                         # 後 16 個代碼
    codes = np.concatenate([lo, hi])
    return (codes.astype(np.float32) - 8) * np.float32(d)
''',
    "4-5": '''def q4_0_quantize(weights):
    blocks = np.asarray(weights, dtype=np.float32).reshape(-1, 32)      # 每列一個 block
    idx = np.abs(blocks).argmax(axis=1, keepdims=True)
    m = np.take_along_axis(blocks, idx, axis=1)                         # 每個 block 的 m
    d = m / np.float32(-8)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, 0, 1 / d).astype(np.float32)             # 整塊都是 0 時不能除
    codes = np.trunc(blocks * inv + np.float32(8.5))
    codes = np.clip(codes, 0, 15).astype(np.uint8)
    packed = codes[:, :16] | (codes[:, 16:] << 4)                       # (block 數, 16)
    d_bytes = d.astype(np.float16).view(np.uint8)                       # (block 數, 2)
    return np.concatenate([d_bytes, packed], axis=1)                    # (block 數, 18)
''',
    "5-3": '''import struct

def read_gguf_header(path):
    with open(path, "rb") as f:
        raw = f.read(24)
    magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", raw)
    return magic, version, n_tensors, n_kv
''',
}

# 填答題的答案（自動測試用；最後一個提示用完也會顯示）
ANSWERS = {"1-2": 5, "2-2": 6, "3-2": 8, "4-4": "NIBBLE", "5-4": "GGUF", "8-3": "LLAMA"}
