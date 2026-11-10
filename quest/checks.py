"""任務檢查器：拿你的函式去跑測試，然後用人話告訴你哪裡怪怪的。

每個檢查器回傳 (是否過關, [要顯示的訊息...])。
"""
from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

from . import q4


class Oops(Exception):
    """檢查沒過（訊息會直接顯示給同學看）。"""


def _call(fn, *args):
    """呼叫同學的函式；出錯時把錯誤翻譯成比較好懂的話。"""
    if not callable(fn):
        raise Oops("我需要的是一個<b>函式</b>，但你給我的好像不是。確認上面那格有執行過？")
    try:
        out = fn(*args)
    except Oops:
        raise
    except struct.error:
        raise Oops("<code>struct.unpack</code> 的格式字串還不對（或是還沒填）。檔頭是 24 bytes："
                   "4 個字元 + 一個 32-bit 整數 + 兩個 64-bit 整數，而且是 little-endian。") from None
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if "ellipsis" in msg.lower() or _unfilled(fn):
            raise Oops("程式碼裡還有 <code>...</code> 沒填完喔！把每個 <code>...</code> 換成你的答案。") from None
        raise Oops(f"你的函式執行時出錯了：<code>{_esc(msg)}</code>") from None
    if out is Ellipsis or out is None:
        raise Oops("函式回傳了 <code>...</code> 或 <code>None</code> —— 是不是還有地方沒填、或忘了 <code>return</code>？")
    return out


def _unfilled(fn) -> bool:
    """函式裡是不是還留著 ... 這種待填的空格？（看編譯後的常數表，裡面有 Ellipsis 就是還沒填完）"""
    try:
        return Ellipsis in fn.__code__.co_consts
    except Exception:  # noqa: BLE001
        return False


def _esc(s) -> str:
    import html
    return html.escape(str(s))


def _has_ellipsis(obj) -> bool:
    if obj is Ellipsis:
        return True
    if isinstance(obj, (tuple, list)):
        return any(_has_ellipsis(o) for o in obj)
    return False


def _arr(x, what="回傳值"):
    if _has_ellipsis(x):
        raise Oops("回傳值裡還有 <code>...</code>，表示還有地方沒填完。")
    try:
        a = np.asarray(x)
    except Exception:  # noqa: BLE001
        raise Oops(f"{what}沒辦法轉成 numpy 陣列。") from None
    if a.dtype == object:
        raise Oops(f"{what}裡面混了奇怪的東西（是不是還有 <code>...</code> 沒填？）")
    return a


# =================================================================== 第 1 章
def check_1_1(fn):
    smol = 134_515_008
    cases = [(smol, 16), (1_000_000, 8), (smol, 4.5), (8_000_000_000, 32)]
    for n, b in cases:
        got = float(_call(fn, n, b))
        want = n * b / 8 / 1e6
        if abs(got - want) <= 1e-6 * max(1.0, want):
            continue
        if abs(got - n * b / 8 / 1024 / 1024) <= 1e-6 * want:
            raise Oops("很接近了！但你用的是 1024 × 1024。這門課我們統一用 <b>1 MB = 1,000,000 bytes</b>（跟檔案總管、Hugging Face 顯示的一樣）。")
        if abs(got - n * b / 1e6) <= 1e-6 * want * 8:
            raise Oops("數字剛好大了 8 倍 —— <b>bit 要先除以 8 才是 byte</b> 喔。")
        if abs(got - n * b / 8) <= 1e-3 * want * 1e6:
            raise Oops("你算出來的是 bytes，還要再除以 1,000,000 才是 MB。")
        raise Oops(f"<code>model_size_mb({n:,}, {b})</code> 應該是 <code>{want:,.3f}</code>，但你的函式回傳 <code>{got:,.3f}</code>。")
    return [f"小史莫用 16 bit 存：<b>{smol * 16 / 8 / 1e6:.1f} MB</b>。跟他說的 269 MB 一模一樣！"]


# =================================================================== 第 2 章
def check_2_1(quantize, dequantize):
    scale = 0.25
    x = np.array([0.0, 0.09, -0.09, 0.2, 0.3, -0.7, 1.0, -1.0, 1.74, 5.0, -5.0], dtype=np.float32)
    want = q4.quantize(x, scale)
    got = _arr(_call(quantize, x.copy(), scale), "quantize 的回傳值")
    if got.shape != want.shape:
        raise Oops(f"quantize 應該回傳跟輸入一樣形狀的陣列 {want.shape}，但拿到 {got.shape}。")
    if not np.allclose(got, np.round(got)):
        raise Oops("quantize 的結果應該全部是<b>整數</b>，但裡面還有小數。是不是忘了 <code>np.round</code>？")
    if got.max() > 7 or got.min() < -8:
        raise Oops(f"4-bit 的代碼只能在 <b>-8 ~ 7</b> 之間，但你的結果出現了 {int(got.min())} ~ {int(got.max())}。"
                   "用 <code>np.clip(q, -8, 7)</code> 把超出去的夾回來。")
    if not np.array_equal(got.astype(np.int64), want.astype(np.int64)):
        floor = np.clip(np.floor(x / scale), -8, 7)
        trunc = np.clip(np.trunc(x / scale), -8, 7)
        if np.array_equal(got, floor) or np.array_equal(got, trunc):
            raise Oops("你好像是用「無條件捨去」（<code>int()</code> / <code>np.floor</code>）。量化要的是<b>四捨五入到最近的格子</b>：<code>np.round</code>。")
        i = int(np.argmax(got.astype(np.int64) != want.astype(np.int64)))
        raise Oops(f"x = {x[i]:g}、scale = {scale} 時，代碼應該是 <code>{int(want[i])}</code>"
                   f"（{x[i]:g} ÷ {scale} = {x[i] / scale:g}，四捨五入再夾進 -8~7），但你的是 <code>{int(got[i])}</code>。")
    q = np.array([-8, -3, 0, 2, 7])
    back = _arr(_call(dequantize, q, scale), "dequantize 的回傳值").astype(np.float64)
    if not np.allclose(back, q * scale, atol=1e-6):
        raise Oops(f"dequantize 很單純：<b>代碼 × scale</b>。例如代碼 2、scale {scale} → {2 * scale}，但你的回傳 {back[3]:g}。")
    rt = _arr(_call(dequantize, _call(quantize, x.copy(), scale), scale)).astype(np.float64)
    err = np.abs(rt - x)[:9].max()
    return [f"測試資料來回一趟（量化 → 還原），範圍內的最大誤差是 <b>{err:.3f}</b>，果然不超過半格（{scale / 2}）。",
            "超出範圍的 5.0 被夾成 7 × 0.25 = 1.75 —— 這種誤差叫做「截斷（clipping）」。"]


# =================================================================== 第 3 章
def check_3_1(absmax_scale, blockwise_roundtrip):
    v = np.array([0.1, -0.7, 0.35, 0.0], dtype=np.float32)
    s = float(_call(absmax_scale, v))
    if abs(s - 0.1) > 1e-6:
        if abs(s - 0.7) < 1e-6:
            raise Oops("你回傳的是最大絕對值本身（0.7）。scale 要讓它<b>剛好對到代碼 7</b>，所以還要除以 7。")
        if abs(s + 0.1) < 1e-6 or abs(s - 0.05) < 1e-6:
            raise Oops("要取<b>絕對值</b>最大的那個：<code>np.max(np.abs(block))</code>，再除以 7。")
        if abs(s - 0.7 / 8) < 1e-6:
            raise Oops("這一章我們先用對稱的 -7 ~ +7，所以是除以 <b>7</b>（除以 8 是下一章 Q4_0 的招式）。")
        raise Oops(f"<code>absmax_scale([0.1, -0.7, 0.35, 0.0])</code> 應該是 0.7 ÷ 7 = <code>0.1</code>，但你的是 <code>{s:g}</code>。")
    rng = np.random.default_rng(3)
    x = (rng.standard_normal(128) * 0.1).astype(np.float32)
    x[77] = 2.5
    for bs in (128, 32, 8):
        got = _arr(_call(blockwise_roundtrip, x.copy(), bs), "blockwise_roundtrip 的回傳值").astype(np.float64)
        want = q4.blockwise_roundtrip(x, bs).astype(np.float64)
        if got.shape != want.shape:
            raise Oops(f"回傳的陣列形狀應該跟輸入一樣 {want.shape}，但拿到 {got.shape}。")
        if not np.allclose(got, want, atol=1e-5):
            whole = q4.blockwise_roundtrip(x, len(x)).astype(np.float64)
            if bs != 128 and np.allclose(got, whole, atol=1e-5):
                raise Oops("你的結果跟「整包只用一個 scale」一模一樣 —— scale 要用<b>每個 block 自己的</b> <code>absmax_scale(block)</code> 來算。")
            if np.allclose(got, x, atol=1e-7):
                raise Oops("回傳值跟輸入完全一樣，看起來沒有真的量化。記得：量化（quantize）→ 還原（dequantize）。")
            i = int(np.argmax(np.abs(got - want)))
            raise Oops(f"block 大小 {bs} 時，第 {i} 個權重（原值 {x[i]:.4f}）應該變成 <code>{want[i]:.4f}</code>，但你的是 <code>{got[i]:.4f}</code>。")
    rows = []
    for bs in (128, 32, 8):
        e = float(np.abs(q4.blockwise_roundtrip(x, bs) - x).mean())
        rows.append(f"block = {bs:>3}：平均誤差 <b>{e:.4f}</b>")
    return ["同一批有老鼠屎的權重，分塊越細誤差越小：　" + "　｜　".join(rows)]


# =================================================================== 第 4 章
def _blocks():
    rng = np.random.default_rng(42)
    a = (rng.standard_normal(32) * 0.05).astype(np.float32)          # 普通的 block
    b = a.copy(); b[5] = 0.31                                         # 最大值是正的 → d 是負的
    c = a.copy(); c[9] = -0.4; c[20] = 0.395                          # 會需要夾住（clip）
    return {"普通的 block": a, "最大值是正數的 block": b, "需要夾住的 block": c}


def _norm_encode(out):
    if _has_ellipsis(out):
        raise Oops("回傳值裡還有 <code>...</code>，表示還有地方沒填完。")
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        raise Oops("q4_0_encode 要回傳兩個東西：<code>return d, codes</code>。")
    d, codes = out
    try:
        d = float(d)
    except Exception:  # noqa: BLE001
        raise Oops("d 應該是一個數字（scale）。") from None
    codes = _arr(codes, "codes")
    if codes.shape != (32,):
        raise Oops(f"codes 應該剛好有 32 個（每個權重一個），但拿到的形狀是 {codes.shape}。")
    return d, codes


def check_4_1(encode):
    for name, blk in _blocks().items():
        want_d, want_c = q4.q4_0_encode(blk)
        d, codes = _norm_encode(_call(encode, blk.copy()))
        if not np.isfinite(d):
            raise Oops("d 算出來是 nan 或 inf，檢查一下除法。")
        if abs(d - want_d) > 2e-3 * abs(want_d):
            if abs(d + want_d) <= 2e-3 * abs(want_d):
                raise Oops(f"【{name}】d 的<b>正負號</b>反了。m 要保留正負號（提示：<code>block[np.argmax(np.abs(block))]</code>），"
                           "而且是除以 <b>-8</b>（負的！）。")
            if abs(abs(d) - abs(want_d) * 8 / 7) <= 2e-3 * abs(want_d):
                raise Oops("你除的是 7。Q4_0 的小聰明是除以 <b>-8</b>，把 -8 那一格也用上。")
            raise Oops(f"【{name}】d 應該是 <code>{want_d:.6f}</code>（m ÷ -8），但你的是 <code>{d:.6f}</code>。")
        if not np.allclose(codes, np.round(codes)):
            raise Oops("codes 應該全部是整數。記得用 <code>np.trunc</code> 去掉小數。")
        if codes.min() < 0 or codes.max() > 15:
            hint = "是不是忘了 <b>+ 8.5</b>？" if codes.min() < 0 else "用 <code>np.clip(codes, 0, 15)</code> 夾住。"
            raise Oops(f"【{name}】代碼必須在 <b>0 ~ 15</b> 之間，但你的範圍是 {int(codes.min())} ~ {int(codes.max())}。{hint}")
        diff = np.abs(codes.astype(np.int64) - want_c.astype(np.int64))
        if (diff > 0).sum() > 1 or diff.max() > 1:
            i = int(np.argmax(diff))
            x = float(blk[i])
            raise Oops(f"【{name}】第 {i} 個權重 x = {x:.5f}：x ÷ d + 8.5 = {x / want_d + 8.5:.3f}，去掉小數應該是代碼 "
                       f"<code>{int(want_c[i])}</code>，但你的是 <code>{int(codes[i])}</code>。")
    d, codes = q4.q4_0_encode(_blocks()["最大值是正數的 block"])
    return [f"注意到了嗎？當最大的權重是正數時，<b>d 是負的</b>（這次是 {d:.5f}）。正負號藏在 scale 裡，於是 16 個格子一格都沒浪費。"]


def _to_bytes(out, what):
    if _has_ellipsis(out):
        raise Oops("回傳值裡還有 <code>...</code>，表示還有地方沒填完。")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    a = _arr(out, what)
    if a.dtype != np.uint8:
        raise Oops(f"{what}應該是 <code>bytes</code>（或 uint8 陣列），但拿到的型別是 <code>{a.dtype}</code>。")
    return a.tobytes()


def check_4_2(pack):
    for name, blk in _blocks().items():
        d, codes = q4.q4_0_encode(blk)
        want = q4.q4_0_pack(d, codes)
        got = _to_bytes(_call(pack, d, codes.copy()), "q4_0_pack 的回傳值")
        if len(got) != 18:
            if len(got) == 20:
                raise Oops("你的結果是 20 bytes：d 好像被存成 float32（4 bytes）了。要用 <code>np.float16(d).tobytes()</code>（2 bytes）。")
            if len(got) == 34:
                raise Oops("你的結果是 34 bytes：32 個代碼各佔了 1 個 byte。要把<b>兩個代碼擠進同一個 byte</b>才會是 16 bytes。")
            raise Oops(f"一個 Q4_0 block 應該剛好 <b>18 bytes</b>（2 + 16），但你的是 {len(got)} bytes。")
        if got[:2] != want[:2]:
            raise Oops(f"前 2 個 byte 應該是 float16 的 d：<code>{want[:2].hex(' ')}</code>，但你的是 <code>{got[:2].hex(' ')}</code>。")
        if got[2:] != want[2:]:
            lo, hi = codes[:16], codes[16:]
            if got[2:] == ((lo << 4) | hi).astype(np.uint8).tobytes():
                raise Oops("高低位放反了！<b>前 16 個</b>代碼放低 4 bit，<b>後 16 個</b>要左移 4 位（<code>&lt;&lt; 4</code>）放高 4 bit。")
            if got[2:] == (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8).tobytes():
                raise Oops("配對方式不對：不是「相鄰兩個」一組，而是<b>第 j 個</b>跟<b>第 j+16 個</b>一組。")
            j = next(k for k in range(16) if got[2 + k] != want[2 + k])
            raise Oops(f"【{name}】第 {j} 個代碼 byte 應該是 <code>{want[2 + j]:02X}</code>"
                       f"（低位 = codes[{j}] = {int(lo[j]):X}，高位 = codes[{j + 16}] = {int(hi[j]):X}），但你的是 <code>{got[2 + j]:02X}</code>。")
    blk = _blocks()["普通的 block"]
    raw = q4.q4_0_pack(*q4.q4_0_encode(blk))
    return [f"32 個 float32 原本要 128 bytes，現在只要 18 bytes：<code>{raw.hex(' ').upper()}</code>"]


def check_4_3(decode):
    worst = 0.0
    for name, blk in _blocks().items():
        raw = q4.q4_0_pack(*q4.q4_0_encode(blk))
        want = q4.q4_0_decode(raw).astype(np.float64)
        got = _arr(_call(decode, raw), "q4_0_decode 的回傳值").astype(np.float64).reshape(-1)
        if got.shape != (32,):
            raise Oops(f"應該解出 32 個權重，但你的結果有 {got.size} 個。")
        if not np.allclose(got, want, atol=1e-6):
            d, codes = q4.q4_0_unpack(raw)
            if np.allclose(got, np.concatenate([want[16:], want[:16]]), atol=1e-6):
                raise Oops("前半和後半接反了：<b>低 4 bit</b>（<code>&amp; 0x0F</code>）是前 16 個，<b>高 4 bit</b>（<code>&gt;&gt; 4</code>）是後 16 個。")
            if np.allclose(got, codes.astype(np.float64) * d, atol=1e-6):
                raise Oops("忘了 <b>減 8</b>：代碼是 0~15，要先減 8 變回 -8~7，再乘上 d。")
            i = int(np.argmax(np.abs(got - want)))
            raise Oops(f"【{name}】第 {i} 個權重應該還原成 <code>{want[i]:.5f}</code>"
                       f"（(代碼 {int(codes[i])} − 8) × d {d:.5f}），但你的是 <code>{got[i]:.5f}</code>。")
        worst = max(worst, float(np.abs(got - blk).max() / np.abs(blk).max()))
    return [f"解碼成功！還原後的誤差最多是該 block 最大值的 <b>{worst * 100:.1f}%</b> —— 不完美，但已經很像了。"]


def check_4_5(quantize):
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((48, 96)) * 0.04).astype(np.float32)
    w[3, :32] = 0                                   # 整塊都是 0 的特殊情況
    want = q4.q4_0_quantize(w)
    got = _arr(_call(quantize, w.copy()), "q4_0_quantize 的回傳值")
    if got.dtype != np.uint8:
        raise Oops(f"回傳值的 dtype 應該是 <code>uint8</code>（一格就是一個 byte），但你的是 <code>{got.dtype}</code>。")
    if got.shape != want.shape:
        raise Oops(f"{w.size} 個權重 = {w.size // 32} 個 block，回傳的形狀應該是 <code>{want.shape}</code>，但你的是 <code>{got.shape}</code>。")
    same = float((got == want).mean())
    if same < 0.999:
        bad = int(np.argmax((got != want).any(axis=1)))
        col = int(np.argmax(got[bad] != want[bad]))
        where = "d（前 2 個 byte）" if col < 2 else f"第 {col - 2} 個代碼 byte"
        if bad == 3 * 3:
            raise Oops("整塊都是 0 的 block 出問題了（d = 0 時不能拿來除）。照著範例裡 <code>np.where(d == 0, 0, 1 / d)</code> 的寫法處理。")
        raise Oops(f"第 {bad} 個 block 的{where}跟教授的版本不一樣（整體只有 {same * 100:.1f}% 相同）。"
                   "對照你在 4-1、4-2 寫的單一 block 版本，每一步應該一模一樣，只是多了一個維度。")
    big = (rng.standard_normal(32 * 60_000) * 0.04).astype(np.float32)
    t = time.perf_counter()
    _call(quantize, big)
    sec = time.perf_counter() - t
    speed = big.size / max(sec, 1e-9) / 1e6
    lines = [f"跟教授的版本 <b>{same * 100:.2f}%</b> 的 byte 一模一樣。",
             f"速度：每秒量化 <b>{speed:.0f} 百萬</b>個權重 —— 小史莫的 1.35 億個權重大約 {134.5 / speed:.0f} 秒就能做完。"]
    if speed < 2:
        lines.append("（有點慢，可能還有 Python 迴圈。不過正確最重要，可以繼續！）")
    return lines


# =================================================================== 第 5 章
def read_header(path) -> dict:
    with open(path, "rb") as f:
        magic, version, n_tensors, n_kv = struct.unpack("<4sIQQ", f.read(24))
    return {"magic": magic, "version": version, "n_tensors": n_tensors, "n_kv": n_kv}


def check_5_3(fn, path):
    path = Path(path)
    if not path.exists():
        raise Oops(f"找不到 <code>{_esc(path)}</code>。先完成任務 5-2（轉檔）再回來。")
    want = read_header(path)
    out = _call(fn, path)
    if _has_ellipsis(out):
        raise Oops("回傳值裡還有 <code>...</code>，表示還有地方沒填完。")
    if isinstance(out, dict):
        vals = [out.get("magic"), out.get("version"), out.get("n_tensors"), out.get("n_kv")]
    elif isinstance(out, (tuple, list)) and len(out) == 4:
        vals = list(out)
    else:
        raise Oops("請回傳 4 個值：<code>return magic, version, n_tensors, n_kv</code>。")
    magic = vals[0]
    if isinstance(magic, str):
        magic = magic.encode("ascii", "replace")
    if magic != want["magic"]:
        raise Oops(f"magic 應該是 <code>{want['magic']!r}</code>，但你讀到 <code>{vals[0]!r}</code>。"
                   "檔頭的格式是 <code>&lt;4sIQQ</code>：4 個字元、一個 32-bit 整數、兩個 64-bit 整數。")
    names = ["version（版本）", "n_tensors（張量數量）", "n_kv（設定數量）"]
    keys = ["version", "n_tensors", "n_kv"]
    for nm, k, v in zip(names, keys, vals[1:]):
        try:
            v = int(v)
        except Exception:  # noqa: BLE001
            raise Oops(f"{nm} 應該是整數。") from None
        if v != want[k]:
            extra = ""
            if k != "version" and v > 10 ** 9:
                extra = " 數字大得離譜，通常是把 64-bit（<code>Q</code>）寫成 32-bit（<code>I</code>），或是 byte 順序（<code>&lt;</code>）錯了。"
            raise Oops(f"{nm} 應該是 <code>{want[k]}</code>，但你讀到 <code>{v}</code>。{extra}")
    return [f"你只用了 24 個 byte 就看穿這個檔案：GGUF 第 <b>{want['version']}</b> 版，"
            f"裡面有 <b>{want['n_tensors']}</b> 個張量、<b>{want['n_kv']}</b> 筆設定。"]


# =================================================================== 填答題
def _num(v):
    if _has_ellipsis(v) or v is None:
        raise Oops("還沒填答案喔！把 <code>...</code> 換成你的答案。")
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        raise Oops("這題的答案是一個數字。") from None


def answer_1_2(v):
    v = _num(v)
    if v == 5:
        return ["5 bit → 84.1 MB，過關！6 bit 就會變成 100.9 MB，剛好超重一點點。",
                "所以我們需要比 6 bit 更省的方法。4-bit 量化，就決定是你了！"]
    if v > 5:
        raise Oops(f"{v:g} bit 會變成 {134_515_008 * v / 8 / 1e6:.1f} MB，超過 100 MB 了。再少一點？")
    raise Oops(f"{v:g} bit 當然過得了橋（{134_515_008 * v / 8 / 1e6:.1f} MB），但題目問的是<b>最多</b>可以用幾個 bit。")


def answer_2_2(v):
    v = _num(v)
    if v == 6:
        return ["6 bit（64 格）→ 最大誤差 0.0161。5 bit 的 0.0333 還差一點。",
                "每多 1 個 bit，誤差就砍半 —— 反過來說，每省 1 個 bit，誤差就加倍。量化就是在這兩者之間討價還價。"]
    if v < 6:
        raise Oops(f"{v:g} bit 的最大誤差還大於 0.02。把滑桿再往右拉。")
    raise Oops(f"{v:g} bit 的確夠了，但題目問的是<b>最少</b>需要幾個。")


def answer_3_2(v):
    v = _num(v)
    if abs(v - 8) < 1e-9:
        return ["4 bit 代碼 + 16 ÷ 4 = 4 bit 的 scale = 8 bit。切這麼細的話，乾脆直接用 8-bit 量化還比較划算！",
                "Q4_0 選的是每 <b>32</b> 個一塊：4 + 16 ÷ 32 = <b>4.5 bit</b>。這個數字等一下會一直出現。"]
    if abs(v - 4) < 1e-9:
        raise Oops("4 bit 只算到代碼本身。每一塊還要多存一個 <b>16-bit 的 scale</b>，由這一塊的 4 個權重一起分攤。")
    if abs(v - 20) < 1e-9:
        raise Oops("20 是「一整塊」的總 bit 數嗎？題目問的是<b>每個權重平均</b>。")
    raise Oops("觀察規律：64 個一塊是 4.25、32 個一塊是 4.5、8 個一塊是 6。公式是 <b>4 + 16 ÷ 每塊的權重數</b>。")


def _word(v):
    if _has_ellipsis(v) or v is None:
        raise Oops("還沒填答案喔！把 <code>...</code> 換成你的答案（記得加引號）。")
    word = str(v).strip().upper()
    if word in ("...", "…", ""):
        raise Oops("還沒填答案喔！把引號裡的 <code>...</code> 換成你的答案。")
    return word


def answer_4_4(v):
    if _word(v) == "NIBBLE":
        return ["Nibble：半個 byte，也就是 4 個 bit。你剛剛親手當了五次 Q4_0 量化器！"]
    raise Oops("密語不對。玩「人肉量化機」答對 5 題，畫面下方會出現通關密語。")


def answer_5_4(v):
    if _word(v) == "GGUF":
        return ["47='G'、47='G'、55='U'、46='F'。每個 GGUF 檔案一開頭就在大聲喊自己的名字。",
                "（咕咕夫教授：「就說是巧合了。」）"]
    raise Oops("再對照一次 ASCII 表：47、47、55、46 各是哪個字母？")


def answer_8_3(v):
    if _word(v) == "LLAMA":
        return ["滿分！你已經具備一位量化師該有的全部觀念了。"]
    raise Oops("密語不對。畢業考全部答對後，畫面下方會出現通關密語。")


CODE_CHECKS = {
    "1-1": check_1_1,
    "2-1": check_2_1,
    "3-1": check_3_1,
    "4-1": check_4_1,
    "4-2": check_4_2,
    "4-3": check_4_3,
    "4-5": check_4_5,
    "5-3": check_5_3,
}
ANSWER_CHECKS = {
    "1-2": answer_1_2,
    "2-2": answer_2_2,
    "3-2": answer_3_2,
    "4-4": answer_4_4,
    "5-4": answer_5_4,
    "8-3": answer_8_3,
}
