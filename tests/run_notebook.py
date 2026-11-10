#!/usr/bin/env python3
"""教材自動測試：把解答填進 Notebook，從頭到尾真的跑一遍。

用法（在專案根目錄）：

    pip install nbclient nbformat ipykernel
    python tests/run_notebook.py                  # 在暫存資料夾裡跑（會下載約 270 MB 的模型）
    python tests/run_notebook.py --workdir /tmp/qc   # 指定資料夾，重跑時可以沿用已下載的檔案

它會：
  1. 把每個 🎯 任務格換成 quest/hints.py 裡的解答
  2. 用 nbclient 執行整本 Notebook
  3. 確認沒有任何一格出錯、而且 23 個任務全部過關

老師改完教材之後跑一次，就知道有沒有不小心改壞。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = "quantize_classroom.ipynb"


def fill_in(nb, hints) -> int:
    """把任務格換成解答。回傳換了幾格。"""
    n = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if cell.id == "boot":
            cell.source = cell.source.replace("在這裡寫上你的名字", "自動測試")
        tags = cell.metadata.get("tags", [])
        tid = next((t.split(":", 1)[1] for t in tags if t.startswith("task:")), None)
        if tid is None:
            continue
        if tid in hints.SOLUTIONS:
            lines = cell.source.splitlines()
            start = next(i for i, ln in enumerate(lines) if ln.startswith("quest.check("))
            cell.source = hints.SOLUTIONS[tid] + "\n" + "\n".join(lines[start:])
        elif tid in hints.ANSWERS:
            cell.source = f"quest.answer({tid!r}, {hints.ANSWERS[tid]!r})"
        else:
            raise SystemExit(f"任務 {tid} 沒有解答")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, help="在這個資料夾裡執行（預設：暫存資料夾，跑完就刪）")
    ap.add_argument("--timeout", type=int, default=1800, help="每一格最多跑幾秒")
    args = ap.parse_args()

    import nbformat
    from nbclient import NotebookClient

    sys.path.insert(0, str(ROOT))
    from quest import game, hints

    work = args.workdir or Path(tempfile.mkdtemp(prefix="quantize-classroom-"))
    work.mkdir(parents=True, exist_ok=True)
    for name in ("quest", "requirements.txt", NOTEBOOK):
        src, dst = ROOT / name, work / name
        if src.is_dir():
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(src, dst)
    (work / "quest_save.json").unlink(missing_ok=True)

    nb = nbformat.read(work / NOTEBOOK, as_version=4)
    n = fill_in(nb, hints)
    print(f"已填入 {n} 個任務的解答，開始執行（工作資料夾：{work}）……")

    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3", allow_errors=True,
                            resources={"metadata": {"path": str(work)}})
    client.execute()
    nbformat.write(nb, work / "executed.ipynb")

    errors = []
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append((i, cell.id, out.get("ename"), out.get("evalue")))
    for i, cid, ename, evalue in errors:
        print(f"✘ 第 {i} 格（{cid}）出錯：{ename}: {str(evalue)[:300]}")

    save = json.loads((work / "quest_save.json").read_text(encoding="utf-8"))
    missing = [t for t in game.TASKS if t not in save["done"]]
    print(f"經驗值 {save['xp']} / {game.TOTAL_XP}；完成 {len(save['done'])} / {len(game.TASKS)} 個任務")
    if missing:
        print("✘ 沒過關的任務：", "、".join(missing))
    stats = save.get("stats", {})
    print("統計：", json.dumps(stats, ensure_ascii=False))
    print(f"執行後的 Notebook：{work / 'executed.ipynb'}")

    ok = not errors and not missing
    print("✔ 全部通過！" if ok else "✘ 有問題，請看上面的訊息。")
    if ok and args.workdir is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
