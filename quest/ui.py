"""畫面：把 HTML / CSS / JavaScript 小遊戲塞進 Notebook 的輸出格。

互動元件都放在 <iframe srcdoc> 裡，所以不管你用 JupyterLab、VS Code
還是 Colab，樣式都不會跟 Notebook 本身打架。
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

from . import game

WEB = Path(__file__).parent / "web"

# 跟 web/base.css 同一組顏色（給不放 iframe 的小卡片用）
C = {
    "bg": "#12172b", "panel": "#1b2240", "panel2": "#252e57", "line": "#36407a",
    "ink": "#f1f3ff", "muted": "#a3abd6", "gold": "#ffd166", "pink": "#ff7aa8",
    "mint": "#5ee6c0", "sky": "#74b9ff", "orange": "#ff9f5a", "red": "#ff6b6b",
}
FONT = ("'PingFang TC','Microsoft JhengHei','Noto Sans TC','Noto Sans CJK TC',"
        "'Heiti TC',system-ui,-apple-system,'Segoe UI',sans-serif")
MONO = "ui-monospace,'SF Mono','Cascadia Mono',Consolas,'Noto Sans Mono',monospace"


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:  # noqa: BLE001
        return False


def _strip(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def _display(html_text: str, plain: str = "") -> None:
    plain = _strip(plain) or "（請在 Jupyter 裡執行才看得到畫面）"
    if not in_notebook():
        print(plain)
        return
    from IPython.display import display
    display({"text/html": html_text, "text/plain": plain}, raw=True)


def esc(text) -> str:
    return html.escape(str(text), quote=True)


# ------------------------------------------------------------------ iframe 元件
def _read(name: str) -> str:
    return (WEB / name).read_text(encoding="utf-8")


def build_page(fragment: str, data: dict | None = None) -> str:
    """把 base.css + base.js + 小遊戲片段組成一份完整的 HTML 文件。"""
    payload = json.dumps(data or {}, ensure_ascii=False).replace("</", "<\\/")
    return (
        "<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<style>{_read('base.css')}</style></head><body>"
        f"<script>window.DATA = {payload};</script>"
        f"<script>{_read('base.js')}</script>"
        f"{_read(fragment + '.html')}"
        "</body></html>"
    )


def iframe(doc: str, height: int) -> str:
    return (
        f'<iframe srcdoc="{html.escape(doc, quote=True)}" '
        f'style="width:100%;height:{int(height)}px;border:0;border-radius:16px;display:block;'
        f'background:{C["bg"]}" loading="lazy" title="量化教室互動元件"></iframe>'
    )


def show_page(fragment: str, data: dict | None = None, height: int = 480, plain: str = "") -> None:
    _display(iframe(build_page(fragment, data), height), plain or f"[互動元件：{fragment}]")


# ------------------------------------------------------------------ 小卡片（不需要 JavaScript）
_CODE = (f'<code style="font-family:{MONO};background:{C["bg"]};color:{C["gold"]};border:1px solid {C["line"]};'
         f'border-radius:6px;padding:1px 6px;font-size:13px">')


def _card(body: str, accent: str) -> str:
    body = body.replace("<code>", _CODE)      # Jupyter 的佈景主題會幫 <code> 加淺色底，這裡統一成我們的樣式
    return (
        f'<div style="font-family:{FONT};background:{C["panel"]};color:{C["ink"]};'
        f'border:2px solid {C["line"]};border-left:8px solid {accent};border-radius:14px;'
        f'padding:14px 18px;margin:6px 0;line-height:1.7;font-size:15px;max-width:920px">{body}</div>'
    )


def _pill(text: str, color: str) -> str:
    return (f'<span style="display:inline-block;background:{color};color:{C["bg"]};font-weight:800;'
            f'border-radius:999px;padding:1px 12px;margin-right:8px;font-size:13px">{text}</span>')


def xp_bar() -> str:
    st = game.state
    lv = game.level_info()
    pct = min(100.0, 100.0 * st["xp"] / game.TOTAL_XP)
    nxt = f'　距離下一級還差 <b>{lv["next"] - st["xp"]}</b> XP' if lv["next"] else "　已達最高等級！"
    badges = "".join(
        f'<span title="{esc(c["badge_name"])}" style="font-size:22px;margin-right:4px;'
        f'{"" if game.chapter_done(ch) else "filter:grayscale(1);opacity:.25;"}">{c["badge"]}</span>'
        for ch, c in game.CHAPTERS.items()
    )
    return (
        f'<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px">'
        f'<div><span style="color:{C["gold"]};font-weight:800">Lv.{lv["level"]} {esc(lv["title"])}</span>'
        f'　{esc(st["name"] or "無名見習生")}</div>'
        f'<div style="flex:1;min-width:200px">'
        f'<div style="background:{C["bg"]};border-radius:999px;height:14px;overflow:hidden;border:1px solid {C["line"]}">'
        f'<div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,{C["mint"]},{C["gold"]})"></div></div>'
        f'<div style="font-size:12px;color:{C["muted"]};margin-top:3px">{st["xp"]} / {game.TOTAL_XP} XP{nxt}</div>'
        f'</div><div>{badges}</div></div>'
    )


def show_hud() -> None:
    lv = game.level_info()
    _display(_card(xp_bar(), C["sky"]), f'Lv.{lv["level"]} {lv["title"]}  {game.state["xp"]}/{game.TOTAL_XP} XP')


def show_success(result: dict, lines: list[str] | None = None) -> None:
    title = esc(result["title"])
    if result["first_time"]:
        head = _pill(f'+{result["gained"]} XP', C["gold"]) + f'<b>任務 {result["task"]}「{title}」完成！</b>'
        if result["solution_seen"]:
            head += f' <span style="color:{C["muted"]};font-size:13px">（看過解答，經驗值減半）</span>'
    else:
        head = _pill("已完成", C["mint"]) + f'<b>任務 {result["task"]}「{title}」再次通過</b>'
    extra = "".join(f'<div style="margin-top:4px">{ln}</div>' for ln in (lines or []))
    if result["level_up"]:
        extra += (f'<div style="margin-top:8px;color:{C["gold"]};font-weight:800">'
                  f'⬆ 升級！你現在是 Lv.{result["level"]["level"]}「{esc(result["level"]["title"])}」</div>')
    if result["new_badge"]:
        ch = game.CHAPTERS[result["chapter"]]
        extra += (f'<div style="margin-top:8px;color:{C["mint"]};font-weight:800">'
                  f'{ch["badge"]} 獲得「{esc(ch["badge_name"])}」— 第 {result["chapter"]} 章全部過關！</div>')
    body = f'<div>{head}</div>{extra}<div style="margin-top:10px">{xp_bar()}</div>'
    _display(_card(body, C["mint"]), f'[過關] 任務 {result["task"]} {result["title"]} +{result["gained"]} XP')


def show_fail(task_id: str, lines: list[str]) -> None:
    title = game.TASKS[task_id][1]
    body = (
        f'<div>{_pill("還差一點", C["orange"])}<b>任務 {task_id}「{esc(title)}」還沒過關</b></div>'
        + "".join(f'<div style="margin-top:4px">{ln}</div>' for ln in lines)
        + f'<div style="margin-top:8px;color:{C["muted"]};font-size:13px">'
          f'🦙 卡住了嗎？執行 <code style="color:{C["gold"]}">quest.hint("{task_id}")</code> 可以看提示（看提示不扣分）。</div>'
    )
    _display(_card(body, C["orange"]), f'[未過關] 任務 {task_id}：' + " / ".join(lines))


def show_note(title: str, lines: list[str], accent: str = "sky", plain: str = "") -> None:
    body = f'<div><b>{title}</b></div>' + "".join(f'<div style="margin-top:4px">{ln}</div>' for ln in lines)
    _display(_card(body, C[accent]), plain or title)


def show_code(title: str, code: str, note: str = "") -> None:
    body = (
        f'<div><b>{title}</b></div>'
        f'<pre style="font-family:{MONO};background:{C["bg"]};color:{C["ink"]};border:1px solid {C["line"]};'
        f'border-radius:10px;padding:12px 14px;margin:8px 0 0;font-size:13px;line-height:1.55;'
        f'overflow-x:auto;white-space:pre">{esc(code)}</pre>'
        + (f'<div style="margin-top:6px;color:{C["muted"]};font-size:13px">{note}</div>' if note else "")
    )
    _display(_card(body, C["gold"]), f"{title}\n{code}")


def show_table(title: str, header: list[str], rows: list[list], note: str = "") -> None:
    """顯示一張表。儲存格可以放 HTML（呼叫的人要自己跳脫特殊字元）。

    刻意不用 <table>：Jupyter / VS Code 的佈景主題會幫表格加上白底斑馬紋，在深色卡片裡會看不見字。
    """
    n = len(header)
    cells = "".join(
        f'<div style="padding:6px 16px 6px 0;color:{C["muted"]};font-weight:600;font-size:13px;'
        f'border-bottom:1px solid {C["line"]}">{esc(h)}</div>' for h in header)
    for row in rows:
        cells += "".join(
            f'<div style="padding:6px 16px 6px 0;border-bottom:1px solid {C["panel2"]};font-family:{MONO};'
            f'font-size:13px;color:{C["ink"]};min-width:0">{c}</div>' for c in row)
    grid = (f'<div style="display:grid;grid-template-columns:repeat({n - 1},max-content) minmax(0,1fr);'
            f'margin-top:8px;overflow-x:auto">{cells}</div>')
    body = (f'<div><b>{title}</b></div>{grid}'
            + (f'<div style="margin-top:8px;color:{C["muted"]};font-size:13px">{note}</div>' if note else ""))
    plain = title + "\n" + "\n".join("  ".join(str(c) for c in r) for r in [header] + rows)
    _display(_card(body, C["sky"]), plain)
