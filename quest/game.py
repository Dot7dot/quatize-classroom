"""遊戲狀態：經驗值、等級、徽章、存檔。"""
from __future__ import annotations

import json
import time
from pathlib import Path

SAVE_FILE = Path("quest_save.json")

# ------------------------------------------------------------------ 資料表
CHAPTERS = {
    0: {"title": "新生報到", "badge": "🎒", "badge_name": "新生徽章"},
    1: {"title": "模型為什麼這麼胖？", "badge": "⚖️", "badge_name": "磅秤徽章"},
    2: {"title": "手搖飲的秘密", "badge": "🧋", "badge_name": "半糖徽章"},
    3: {"title": "一顆老鼠屎", "badge": "🐭", "badge_name": "捕鼠徽章"},
    4: {"title": "Q4_0 解剖課", "badge": "🔬", "badge_name": "解剖徽章"},
    5: {"title": "GGUF 藏寶箱", "badge": "📦", "badge_name": "藏寶徽章"},
    6: {"title": "親手量化", "badge": "🔨", "badge_name": "鍛造徽章"},
    7: {"title": "官方神器", "badge": "🛠️", "badge_name": "神器徽章"},
    8: {"title": "最終決戰", "badge": "👑", "badge_name": "屠魔徽章"},
}

# 任務編號: (章節, 名稱, 經驗值)
TASKS = {
    "0-1": (0, "新生報到", 20),
    "0-2": (0, "裝備檢查", 30),
    "1-1": (1, "幫小史莫量體重", 80),
    "1-2": (1, "過橋需要幾 bit？", 60),
    "2-1": (2, "量化與還原", 100),
    "2-2": (2, "甜度量化機", 60),
    "3-1": (3, "分塊量化", 100),
    "3-2": (3, "老鼠屎實驗", 60),
    "4-1": (4, "Q4_0 編碼", 120),
    "4-2": (4, "打包成 18 bytes", 100),
    "4-3": (4, "解碼還原", 100),
    "4-4": (4, "人肉量化機", 80),
    "4-5": (4, "向量化量化器", 120),
    "5-1": (5, "召喚小史莫", 60),
    "5-2": (5, "轉成 GGUF", 100),
    "5-3": (5, "親手讀檔頭", 100),
    "5-4": (5, "藏寶箱地圖", 60),
    "6-1": (6, "量化第一個張量", 80),
    "6-2": (6, "打造手工 GGUF", 200),
    "7-1": (7, "官方神器 llama-quantize", 120),
    "8-1": (8, "讓小史莫開口說話", 150),
    "8-2": (8, "智商檢定", 120),
    "8-3": (8, "畢業考", 100),
}

# (需要的經驗值, 稱號)
LEVELS = [
    (0, "見習量化師"),
    (200, "四捨五入學徒"),
    (500, "位元工匠"),
    (900, "區塊魔導士"),
    (1400, "GGUF 鍛造師"),
    (1900, "量化大師"),
]

TOTAL_XP = sum(xp for _, _, xp in TASKS.values())


# ------------------------------------------------------------------ 狀態
def _blank() -> dict:
    return {"name": "", "xp": 0, "done": {}, "hints": {}, "stats": {}, "started": time.time()}


state: dict = _blank()


def load() -> dict:
    """讀取存檔（沒有就開新檔）。"""
    global state
    state = _blank()
    if SAVE_FILE.exists():
        try:
            saved = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                state.update(saved)
        except Exception:
            pass  # 存檔壞了就當作新遊戲
    return state


def save() -> None:
    try:
        SAVE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # 唯讀環境也不要讓遊戲中斷


def reset() -> None:
    """清除存檔，重新開始。"""
    global state
    state = _blank()
    if SAVE_FILE.exists():
        SAVE_FILE.unlink()


# ------------------------------------------------------------------ 查詢
def level_info(xp: int | None = None) -> dict:
    xp = state["xp"] if xp is None else xp
    idx = 0
    for i, (need, _) in enumerate(LEVELS):
        if xp >= need:
            idx = i
    need, title = LEVELS[idx]
    nxt = LEVELS[idx + 1][0] if idx + 1 < len(LEVELS) else None
    return {"level": idx + 1, "title": title, "floor": need, "next": nxt}


def is_done(task_id: str) -> bool:
    return task_id in state["done"]


def chapter_done(ch: int) -> bool:
    return all(is_done(t) for t, (c, _, _) in TASKS.items() if c == ch)


def badges() -> list[int]:
    return [ch for ch in CHAPTERS if chapter_done(ch)]


def hints_used(task_id: str) -> int:
    return int(state["hints"].get(task_id, 0))


def use_hint(task_id: str) -> int:
    n = hints_used(task_id) + 1
    state["hints"][task_id] = n
    save()
    return n


def set_stat(key: str, value) -> None:
    state["stats"][key] = value
    save()


def get_stat(key: str, default=None):
    return state["stats"].get(key, default)


# ------------------------------------------------------------------ 給分
def award(task_id: str, solution_seen: bool = False) -> dict:
    """完成任務。回傳這次的結算資訊（給 UI 顯示用）。"""
    ch, title, xp = TASKS[task_id]
    before = level_info()
    had_badge = chapter_done(ch)
    first_time = not is_done(task_id)
    gained = 0
    if first_time:
        gained = xp // 2 if solution_seen else xp
        state["xp"] += gained
        state["done"][task_id] = {"xp": gained, "t": time.time()}
        save()
    after = level_info()
    return {
        "task": task_id,
        "title": title,
        "chapter": ch,
        "gained": gained,
        "first_time": first_time,
        "level_up": after["level"] > before["level"],
        "level": after,
        "new_badge": (not had_badge) and chapter_done(ch),
        "solution_seen": solution_seen,
    }
