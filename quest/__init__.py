"""量化教室的遊戲引擎。

在 Notebook 裡這樣用：

    import quest
    quest.start("你的名字")        # 開始冒險（會自動讀取存檔）
    quest.scene("ch1")             # 播放劇情
    quest.play("backpack")         # 打開互動小遊戲
    quest.check("1-1", 你的函式)    # 檢查程式題
    quest.answer("1-2", 你的答案)   # 回答觀察題
    quest.hint("1-1")              # 看提示
    quest.hud()                    # 看目前的等級、經驗值、徽章
"""
from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import warnings as _warnings
from pathlib import Path as _Path

# 沒裝 ipywidgets 時 tqdm 會碎碎念，跟我們無關，先請它安靜
_warnings.filterwarnings("ignore", message=".*IProgress not found.*")

from . import checks, game, ggufkit, hints, story, toolbox, ui
from . import q4 as prof
from .checks import Oops
from .toolbox import run

tools = toolbox      # quest.tools.xxx：下載、轉檔、找 llama.cpp
kit = ggufkit        # quest.kit.xxx  ：GGUF 檔案的小幫手

__all__ = ["start", "scene", "play", "check", "answer", "hint", "hud", "checkpoint", "talk",
           "boss", "certificate", "reset", "run", "tools", "kit", "prof", "my_or_prof", "note", "load_torch_model"]

_solution_seen: set[str] = set()


# ------------------------------------------------------------------ 開始 / 狀態
def start(name: str | None = None) -> None:
    """開始（或繼續）冒險。name 是要寫在畢業證書上的名字。"""
    game.load()
    if name and name.strip() and "名字" not in name:
        game.state["name"] = name.strip()[:24]
        game.save()
    resumed = game.state["xp"] > 0
    title = (f'歡迎回來，{ui.esc(game.state["name"] or "見習生")}！已讀取存檔。' if resumed
             else f'歡迎入學，{ui.esc(game.state["name"] or "見習生")}！')
    ui._display(
        f'<div style="font-family:{ui.FONT};background:linear-gradient(135deg,#1a2150,#3a2f63);color:{ui.C["ink"]};'
        f'border:2px solid {ui.C["line"]};border-radius:16px;padding:22px 24px;max-width:920px">'
        f'<div style="font-size:13px;letter-spacing:.3em;color:{ui.C["gold"]};font-weight:800">QUANTIZE ACADEMY</div>'
        f'<div style="font-size:30px;font-weight:800;margin:4px 0 2px">量化教室</div>'
        f'<div style="font-size:16px;color:{ui.C["muted"]}">把 AI 塞進口袋的 4-bit 大冒險</div>'
        f'<div style="margin-top:14px;font-size:15px">{title}</div></div>',
        f"量化教室 — {title}")
    if not game.state["name"]:
        ui.show_note("✏️ 還沒寫名字", ['把上面那一格改成 <code>quest.start("你的名字")</code> 再執行一次，名字會印在畢業證書上。'],
                     accent="orange")
    result = game.award("0-1")
    if result["first_time"]:
        ui.show_success(result)
    else:
        ui.show_hud()


def hud() -> None:
    """顯示等級、經驗值、徽章。"""
    ui.show_hud()


def reset(confirm: str = "") -> None:
    """清除存檔重新開始：quest.reset("我確定")"""
    if confirm != "我確定":
        ui.show_note("⚠️ 這會清除所有進度", ['真的要重來的話，請執行 <code>quest.reset("我確定")</code>。'], accent="orange")
        return
    game.reset()
    _solution_seen.clear()
    ui.show_note("🧹 存檔已清除", ["重新執行 <code>quest.start(...)</code> 開始新的冒險。"])


def note(title: str, *lines: str) -> None:
    """顯示一張說明小卡。"""
    ui.show_note(title, list(lines))


# ------------------------------------------------------------------ 劇情 / 小遊戲
def scene(scene_id: str) -> None:
    """播放一段劇情。"""
    sc = story.SCENES[scene_id]
    lines = [{"who": who, "text": text} for who, text in sc["lines"]]
    plain = "\n".join(f"{who}: {text}" for who, text in sc["lines"])
    ui.show_page("scene", {"title": sc["title"], "bg": sc.get("bg", ""), "lines": lines}, height=470, plain=plain)


_WIDGETS = {
    "backpack": 600, "sugar": 720, "outlier": 740, "dissect": 930, "human": 560,
    "ggufmap": 720, "bridge": 560, "boss": 640, "quiz": 900, "certificate": 640,
}


def play(widget: str, **data) -> None:
    """打開一個互動小遊戲。"""
    if widget not in _WIDGETS:
        raise ValueError(f"沒有叫做 {widget!r} 的小遊戲。可以用的有：{', '.join(_WIDGETS)}")
    if widget == "backpack":
        data = {"params": toolbox.N_PARAMS, "limit_mb": toolbox.BRIDGE_LIMIT_MB, **data}
    elif widget == "ggufmap":
        path = data.get("path", toolbox.F16_PATH)
        if not _Path(path).exists():
            ui.show_note("📦 還沒有 GGUF 檔", ["先完成任務 5-2（轉檔），再回來打開藏寶箱地圖。"], accent="orange")
            return
        data = ggufkit.map_data(path)
    elif widget == "quiz":
        data = dict(story.FINAL_EXAM)
    ui.show_page(widget, data, height=_WIDGETS[widget])


# ------------------------------------------------------------------ 檢查 / 回答 / 提示
def _finish(task_id: str, lines: list[str]) -> None:
    result = game.award(task_id, solution_seen=task_id in _solution_seen)
    ui.show_success(result, lines)


def check(task_id: str, *objs) -> None:
    """檢查程式題。例如 quest.check("2-1", quantize, dequantize)"""
    fn = checks.CODE_CHECKS.get(task_id)
    if fn is None:
        raise ValueError(f"任務 {task_id} 不是程式題。")
    try:
        lines = fn(*objs)
    except Oops as e:
        ui.show_fail(task_id, [str(e)])
        return
    except TypeError as e:
        ui.show_fail(task_id, [f"呼叫方式好像不太對：<code>{ui.esc(e)}</code>"])
        return
    _finish(task_id, lines)


def answer(task_id: str, value) -> None:
    """回答觀察題 / 輸入通關密語。例如 quest.answer("1-2", 5)"""
    fn = checks.ANSWER_CHECKS.get(task_id)
    if fn is None:
        raise ValueError(f"任務 {task_id} 不是填答題。")
    try:
        lines = fn(value)
    except Oops as e:
        ui.show_fail(task_id, [str(e)])
        return
    _finish(task_id, lines)


def hint(task_id: str) -> None:
    """看提示。每呼叫一次多看一個；提示用完之後會顯示解答（程式題看解答，經驗值減半）。"""
    if task_id not in game.TASKS:
        raise ValueError(f"沒有任務 {task_id}。")
    tips = hints.HINTS.get(task_id, [])
    n = game.use_hint(task_id)
    title = game.TASKS[task_id][1]
    if n <= len(tips):
        lines = [f"{i + 1}. {t}" for i, t in enumerate(tips[:n])]
        more = ("還想要更多提示就再執行一次。" if n < len(tips)
                else "提示用完了。再執行一次會直接顯示<b>解答</b>" + ("（經驗值減半）。" if task_id in hints.SOLUTIONS else "。"))
        ui.show_note(f"💡 任務 {task_id}「{title}」的提示（{n}/{len(tips)}）", lines + [f'<span style="color:{ui.C["muted"]}">{more}</span>'],
                     accent="gold")
        return
    if task_id in hints.SOLUTIONS:
        if not game.is_done(task_id):
            _solution_seen.add(task_id)
        ui.show_code(f"📖 任務 {task_id}「{title}」的解答", hints.SOLUTIONS[task_id],
                     note="先讀懂每一行在做什麼，再自己打一次（不要複製貼上，手指會幫你記住）。")
    elif task_id in hints.ANSWERS:
        ui.show_note(f"📖 任務 {task_id}「{title}」的答案", [f"<code>{ui.esc(hints.ANSWERS[task_id])}</code>"], accent="gold")
    else:
        ui.show_note(f"💡 任務 {task_id}「{title}」", ["這個任務只要把那一格順利執行完就會過關。看看錯誤訊息怎麼說？"], accent="gold")


def my_or_prof(task_id: str, mine, name: str = ""):
    """已經通過任務就用你自己寫的函式；還沒通過就先借用教授的版本，讓冒險能繼續。"""
    ref = {"4-5": prof.q4_0_quantize, "4-3": prof.q4_0_decode, "2-1": prof.quantize}.get(task_id)
    if callable(mine) and game.is_done(task_id):
        ui.show_note("🔨 使用你自己寫的量化器", [f"任務 {task_id} 已通過 —— 接下來用的是<b>你親手寫的</b> <code>{name or mine.__name__}</code>。"],
                     accent="mint")
        return mine
    ui.show_note("🦙 先借用教授的版本",
                 [f"還沒找到通過任務 {task_id} 的函式（可能是重新開啟 Notebook 後還沒重新執行第 4 章的程式格）。",
                  "先用教授的版本繼續；想用自己的，回去把第 4 章的格子重新執行一次再回來。"], accent="orange")
    return ref


# ------------------------------------------------------------------ 自動過關的任務
def checkpoint(task_id: str, **kw) -> None:
    """實戰章節的檢查點：確認這一步真的做成功了。"""
    fn = _CHECKPOINTS.get(task_id)
    if fn is None:
        raise ValueError(f"任務 {task_id} 沒有檢查點。")
    try:
        lines = fn(**kw)
    except Oops as e:
        ui.show_fail(task_id, [str(e)])
        return
    _finish(task_id, lines)
    after = _AFTER.get(task_id)
    if after:
        after()


def _cp_packages():
    have, missing = toolbox.check_packages()
    if missing:
        raise Oops("還缺少這些套件：<b>" + "、".join(missing) + "</b>。執行上面的安裝格（把 <code>INSTALL = True</code>），裝完後重新啟動 kernel 再來。")
    return ["裝備齊全：" + "、".join(f"{n} {v}" for n, v in have)]


def _cp_model(model_dir=None):
    d = _Path(model_dir or toolbox.MODEL_DIR)
    st = d / "model.safetensors"
    if not st.exists() or not (d / "config.json").exists() or not (d / "tokenizer.json").exists():
        raise Oops(f"在 <code>{ui.esc(d)}</code> 找不到完整的模型檔。下載有成功嗎？（需要網路連線）")
    size = toolbox.mb(st)
    if size < 200:
        raise Oops(f"model.safetensors 只有 {size:.1f} MB，看起來沒有下載完整。刪掉 <code>{ui.esc(d)}</code> 再下載一次。")
    game.set_stat("orig_mb", size)
    return [f"小史莫降臨！權重檔 <code>model.safetensors</code> 有 <b>{size:.1f} MB</b>。"]


def _cp_f16(path=None):
    path = _Path(path or toolbox.F16_PATH)
    if not path.exists():
        raise Oops(f"找不到 <code>{ui.esc(path)}</code>，轉檔沒有成功。往上看看指令輸出了什麼錯誤？")
    try:
        s = ggufkit.summary(path)
    except Exception as e:  # noqa: BLE001
        raise Oops(f"這個檔案讀不起來，可能轉到一半就中斷了：<code>{ui.esc(e)}</code>") from None
    if s["types"].get("F16", 0) == 0:
        raise Oops("檔案裡沒有 F16 的張量，確認轉檔時有加 <code>--outtype f16</code>。")
    game.set_stat("f16_mb", s["size_mb"])
    game.set_stat("n_weights", s["n_weights"])
    return [f"搬家成功！<code>{ui.esc(path.name)}</code>：<b>{s['size_mb']:.1f} MB</b>、{s['n_tensors']} 個張量、"
            f"{s['n_weights']:,} 個權重，平均每個權重 {s['bits_per_weight']:.1f} bit。"]


def _cp_one_tensor(original=None, restored=None):
    import numpy as np
    if original is None or restored is None:
        raise Oops("需要 <code>original</code>（原本的權重）和 <code>restored</code>（量化後還原的權重）。")
    a, b = np.asarray(original, dtype=np.float64), np.asarray(restored, dtype=np.float64)
    if a.shape != b.shape:
        raise Oops(f"兩個陣列的形狀不一樣：{a.shape} vs {b.shape}。")
    rel = float(np.sqrt(((a - b) ** 2).mean()) / np.sqrt((a ** 2).mean()))
    if not np.isfinite(rel) or rel > 0.3:
        raise Oops(f"還原後的誤差太大了（相對誤差 {rel * 100:.1f}%），量化器可能有問題。回第 4 章確認任務 4-5 有通過。")
    if rel < 1e-4:
        raise Oops("誤差幾乎是 0 —— 看起來沒有真的量化。<code>restored</code> 應該是量化後再解回來的權重。")
    return [f"第一個真實的張量量化成功！相對誤差 <b>{rel * 100:.1f}%</b>，大小卻只剩 F16 的 28%。"]


def _cp_handmade(path=None):
    gg = ggufkit._gguf()
    path = _Path(path or toolbox.HANDMADE_PATH)
    if not path.exists():
        raise Oops(f"找不到 <code>{ui.esc(path)}</code>。上面那一格有順利執行完嗎？")
    try:
        r = gg.GGUFReader(str(path))
    except Exception as e:  # noqa: BLE001
        raise Oops(f"檔案讀不起來：<code>{ui.esc(e)}</code>。是不是忘了最後的 <code>writer.close()</code>？") from None
    wrong = [t.name for t in r.tensors if ggufkit.should_quantize(t) and t.tensor_type.name != "Q4_0"]
    if wrong:
        raise Oops(f"有 {len(wrong)} 個應該量化的張量不是 Q4_0（例如 <code>{ui.esc(wrong[0])}</code>）。")
    ft = r.fields.get("general.file_type")
    if ft is None or int(ft.contents()) != int(gg.LlamaFileType.MOSTLY_Q4_0):
        raise Oops("metadata 裡的 <code>general.file_type</code> 應該改成 MOSTLY_Q4_0。")
    del r
    s = ggufkit.summary(path)
    game.set_stat("q4_mb", s["size_mb"])
    game.set_stat("q4_bpw", s["bits_per_weight"])
    before = game.get_stat("f16_mb") or (toolbox.mb(toolbox.F16_PATH) if toolbox.F16_PATH.exists() else 270.9)
    return [f"你親手做出了一個 Q4_0 的 GGUF 檔！<b>{before:.1f} MB → {s['size_mb']:.1f} MB</b>，"
            f"平均每個權重 <b>{s['bits_per_weight']:.2f} bit</b>。",
            "檔案在 <code>" + ui.esc(path) + "</code>。來看看過不過得了橋 ↓"]


def _after_handmade():
    before = game.get_stat("f16_mb") or 270.9
    after = game.get_stat("q4_mb") or 0
    ui.show_page("bridge", {"before_mb": before, "after_mb": after, "limit_mb": toolbox.BRIDGE_LIMIT_MB,
                            "ok": after <= toolbox.BRIDGE_LIMIT_MB}, height=_WIDGETS["bridge"])


def _cp_official(official=None, result=None, match_pct=None):
    if official is not None and _Path(official).exists():
        try:
            s = ggufkit.summary(official)
        except Exception as e:  # noqa: BLE001
            raise Oops(f"官方工具產生的檔案讀不起來：<code>{ui.esc(e)}</code>") from None
        if s["types"].get("Q4_0", 0) == 0:
            raise Oops("官方版的檔案裡沒有 Q4_0 的張量，確認指令最後一個參數是 <code>Q4_0</code>。")
        game.set_stat("official_mb", s["size_mb"])
        pct = (result or {}).get("byte_match_pct")
        if pct is not None:
            game.set_stat("match_pct", pct)
        lines = [f"官方版 <code>{ui.esc(_Path(official).name)}</code>：<b>{s['size_mb']:.1f} MB</b>。你已經會用業界標準的量化工具了！"]
        if pct is not None:
            lines.append(f"而且在型別相同的張量裡，你手寫的量化器跟官方的 C++ 程式有 <b>{pct:.2f}%</b> 的 byte 一模一樣。")
        return lines
    if match_pct is not None:
        if match_pct < 99.0:
            raise Oops(f"跟官方參考實作只有 {match_pct:.2f}% 相同，量化器可能有問題。回第 4 章確認任務 4-5。")
        game.set_stat("match_pct", match_pct)
        return [f"這台電腦沒有 llama-quantize，所以改跟官方 <code>gguf</code> 套件裡的參考實作比對：<b>{match_pct:.2f}%</b> 的 byte 一模一樣！",
                "之後裝好 llama.cpp 再回來執行這一格，就能親手跑一次官方工具。"]
    raise Oops("還沒有官方版的檔案，也沒有比對結果。上面那一格有順利執行完嗎？")


def _cp_talk(answers=None):
    if not answers:
        raise Oops("還沒有任何回答。上面那一格有順利執行完嗎？")
    quant = {k: v for k, v in answers.items() if "Q4_0" in k and v and v.get("text")}
    if not quant:
        raise Oops("量化後的模型沒有成功產生文字。確認第 6 章的手工 GGUF 有做出來。")
    game.set_stat("talked", True)
    return ["量化後的小史莫開口說話了！少了三分之二的體重，還是答得出來。"]


def _cp_ppl(ppl=None):
    import math
    if not ppl or len(ppl) < 2:
        raise Oops("需要至少兩個困惑度（原版和量化版）才能比較。")
    vals = list(ppl.items())
    base_name, base = vals[0]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) and v > 1 for _, v in vals):
        raise Oops("困惑度裡有不合理的數字（應該是大於 1 的有限數值）。")
    lines = []
    for name, v in vals[1:]:
        lines.append(f"{ui.esc(name)}：困惑度 {v:.2f}（比{ui.esc(base_name)}高 <b>{(v / base - 1) * 100:.1f}%</b>）")
    worst = max(v for _, v in vals[1:])
    if worst > base * 3:
        raise Oops(f"量化版的困惑度（{worst:.1f}）比原版（{base:.1f}）高太多了，量化過程可能有錯。")
    game.set_stat("ppl", {k: float(v) for k, v in vals})
    lines.append("困惑度變高代表有一點點變笨 —— 這就是量化的代價。但想想看，檔案只剩不到三分之一！")
    return lines


_CHECKPOINTS = {
    "0-2": _cp_packages, "5-1": _cp_model, "5-2": _cp_f16, "6-1": _cp_one_tensor, "6-2": _cp_handmade,
    "7-1": _cp_official, "8-1": _cp_talk, "8-2": _cp_ppl,
}
_AFTER = {"6-2": _after_handmade}


# ------------------------------------------------------------------ 第 8 章：讓模型說話
_torch_cache: dict = {}


def load_torch_model(model_dir=None):
    """載入原版的 PyTorch 模型和 tokenizer（只載入一次）。"""
    key = str(model_dir or toolbox.MODEL_DIR)
    if key not in _torch_cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:                                    # 關掉進度條，畫面比較乾淨
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
        except Exception:  # noqa: BLE001
            pass
        tok = AutoTokenizer.from_pretrained(key)
        model = AutoModelForCausalLM.from_pretrained(key).float().eval()
        _torch_cache[key] = (model, tok)
    return _torch_cache[key]


def talk(question: str, files: dict, llama=None, max_tokens: int = 80) -> dict:
    """讓每個 GGUF 檔回答同一個問題，並把回答排成一張表。

    有 llama.cpp 就用 llama.cpp 跑（真正吃 GGUF 檔的引擎）；沒有的話，
    把 GGUF 裡的權重裝回 PyTorch 模型來跑（數學上等價，只是比較慢）。
    """
    import time
    out, rows = {}, []
    for label, path in files.items():
        if path is None or not _Path(path).exists():
            continue
        res, engine = None, ""
        if llama is not None:
            res = toolbox.llama_generate(llama, path, question, max_tokens)
            engine = "llama.cpp"
        if res is None:
            try:
                model, tok = load_torch_model()
                ggufkit.load_into_torch(model, path)
                t = time.time()
                text = ggufkit.torch_generate(model, tok, question, max_tokens)
                res = {"text": text, "seconds": time.time() - t, "tokens_per_sec": None}
                engine = "PyTorch"
            except Exception as e:  # noqa: BLE001
                res = {"text": "", "seconds": 0.0, "tokens_per_sec": None}
                engine = f"失敗：{type(e).__name__}"
        res["engine"] = engine
        res["size_mb"] = toolbox.mb(path)
        out[label] = res
        speed = f'{res["tokens_per_sec"]:.0f} tok/s' if res.get("tokens_per_sec") else f'{res["seconds"]:.1f} 秒'
        rows.append([ui.esc(label), f'{res["size_mb"]:.1f} MB', engine, speed,
                     f'<span style="font-family:{ui.FONT};font-size:14px;white-space:pre-wrap">{ui.esc(res["text"]) or "（沒有輸出）"}</span>'])
    ui.show_table(f"💬 問：{ui.esc(question)}", ["模型", "大小", "引擎", "速度", "回答"], rows,
                  note="小史莫只有 1.35 億個參數，主要會說英文，偶爾會講錯事情 —— 這是模型太小的關係，不是量化的錯。重點是比較量化前後的差別。")
    return out


def boss() -> None:
    """最終決戰：把你一路上的成果變成必殺技。"""
    st = game.state["stats"]
    f16 = st.get("f16_mb") or 270.9
    q4mb = st.get("q4_mb")
    hits = []
    if q4mb:
        hits.append({"label": "瘦身斬", "detail": f"{f16:.0f} MB → {q4mb:.1f} MB（只剩 {q4mb / f16 * 100:.0f}%）", "dmg": 30})
        hits.append({"label": "半位元組連擊", "detail": f"每個權重只花 {st.get('q4_bpw', 4.5):.2f} bit", "dmg": 20})
    if st.get("match_pct") is not None:
        hits.append({"label": "分毫不差", "detail": f"跟官方量化器 {st['match_pct']:.2f}% 的 byte 相同", "dmg": 20})
    if st.get("talked"):
        hits.append({"label": "開口說話", "detail": "量化後照樣回答問題", "dmg": 15})
    ppl = st.get("ppl")
    if ppl and len(ppl) >= 2:
        vals = list(ppl.values())
        hits.append({"label": "智商保衛戰", "detail": f"困惑度 {vals[0]:.2f} → {min(vals[1:]):.2f}，沒有變成笨蛋", "dmg": 15})
    if len(hits) < 3:
        ui.show_note("⚔️ 招式還不夠", ["先完成第 6～8 章的任務（做出 GGUF、讓模型說話、做智商檢定），再來挑戰魔王！"], accent="orange")
        return
    ui.show_page("boss", {"hits": hits}, height=_WIDGETS["boss"])


# ------------------------------------------------------------------ 畢業證書
def certificate() -> None:
    """顯示畢業證書（任務還沒全部完成的話，會是「修業證明」）。"""
    st = game.state
    lv = game.level_info()
    missing = [f"{t}「{info[1]}」" for t, info in game.TASKS.items() if not game.is_done(t)]
    stats = st["stats"]
    facts = []
    if stats.get("f16_mb"):
        facts.append(["量化前", f'{stats["f16_mb"]:.1f} MB'])
    if stats.get("q4_mb"):
        facts.append(["量化後", f'{stats["q4_mb"]:.1f} MB'])
        if stats.get("f16_mb"):
            facts.append(["壓縮倍率", f'{stats["f16_mb"] / stats["q4_mb"]:.2f}×'])
    if stats.get("q4_bpw"):
        facts.append(["每權重", f'{stats["q4_bpw"]:.2f} bit'])
    facts.append(["經驗值", f'{st["xp"]} XP'])
    digest = _hashlib.sha256(f'{st["name"]}|{st["xp"]}|{sorted(st["done"])}|{stats.get("q4_mb")}'.encode()).hexdigest()[:10].upper()
    data = {
        "name": st["name"] or "無名見習生", "complete": not missing, "model": toolbox.MODEL_NAME,
        "level": lv["level"], "title": lv["title"], "done": len(st["done"]), "total_tasks": len(game.TASKS),
        "missing": missing[:6] + (["……"] if len(missing) > 6 else []), "facts": facts,
        "badges": [{"icon": c["badge"], "name": c["badge_name"], "on": game.chapter_done(ch)} for ch, c in game.CHAPTERS.items()],
        "code": f"QA-{digest}", "date": _dt.date.today().strftime("%Y 年 %m 月 %d 日"),
    }
    ui.show_page("certificate", data, height=_WIDGETS["certificate"],
                 plain=f'{"畢業證書" if not missing else "修業證明"}：{data["name"]} Lv.{lv["level"]} {lv["title"]} {st["xp"]} XP')


# 讀取存檔（如果有的話），這樣重新開啟 Notebook 後直接從中間繼續也沒問題
game.load()
