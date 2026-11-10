"""工具箱：下載模型、取得 llama.cpp 的工具、執行外部指令。

這裡的東西跟「量化的原理」無關，純粹是幫你把環境弄好的苦工，
所以收在這裡，不佔 Notebook 的版面。想看細節的話歡迎翻閱。
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from . import ui

# ------------------------------------------------------------------ 固定的設定
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_NAME = "SmolLM2-135M-Instruct"
N_PARAMS = 134_515_008
BRIDGE_LIMIT_MB = 100

# 釘住同一個 llama.cpp 版本，這樣每位同學拿到的工具都一樣
LLAMA_TAG = "b12244"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp"

WORK = Path("workspace")
MODEL_DIR = WORK / "models" / MODEL_NAME
SRC_DIR = WORK / "llama.cpp-src"
BIN_DIR = WORK / "llama.cpp-bin"
GGUF_DIR = WORK / "gguf"
LOG_DIR = WORK / "logs"
F16_PATH = GGUF_DIR / "smol-f16.gguf"
HANDMADE_PATH = GGUF_DIR / "smol-Q4_0-handmade.gguf"
OFFICIAL_PATH = GGUF_DIR / "smol-Q4_0.gguf"

MODEL_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]


# Windows 上從 Jupyter 執行主控台程式時，不要彈出黑色視窗
_NO_WINDOW = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {}


def mb(path) -> float:
    """檔案大小（MB，1 MB = 1,000,000 bytes）。"""
    return os.path.getsize(path) / 1e6


def is_valid_gguf(path) -> bool:
    """這個檔案是不是一個讀得起來的 GGUF？"""
    path = Path(path)
    if not path.exists() or path.stat().st_size < 32:
        return False
    try:
        import gguf
        return len(gguf.GGUFReader(str(path)).tensors) > 0
    except Exception:  # noqa: BLE001
        return False


def ensure_dirs() -> None:
    for d in (WORK, GGUF_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ 環境檢查
REQUIRED = [
    ("numpy", "numpy"), ("gguf", "gguf"), ("huggingface_hub", "huggingface_hub"),
    ("torch", "torch"), ("transformers", "transformers"), ("matplotlib", "matplotlib"),
]


def check_packages() -> tuple[list[tuple[str, str]], list[str]]:
    """回傳 ([(套件, 版本)...], [缺少的套件...])。"""
    import importlib
    have, missing = [], []
    for mod, pip_name in REQUIRED:
        try:
            m = importlib.import_module(mod)
            have.append((pip_name, getattr(m, "__version__", "?")))
        except Exception:  # noqa: BLE001
            missing.append(pip_name)
    return have, missing


def pip_install_command() -> list[str]:
    """安裝所需套件的指令（Linux 上特別指定 CPU 版 PyTorch，省下好幾 GB）。"""
    cmd = [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"]
    if platform.system() == "Linux":
        cmd += ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
    return cmd


# ------------------------------------------------------------------ 執行外部指令
def run(cmd, title: str = "", log_name: str = "", tail: int = 14, env: dict | None = None,
        timeout: float | None = None, quiet: bool = False) -> bool:
    """執行一個指令，把最後幾行輸出即時顯示出來；完整輸出存到 workspace/logs/。

    回傳 True 代表指令成功（exit code 0）。
    """
    ensure_dirs()
    cmd = [str(c) for c in cmd]
    log_path = LOG_DIR / ((log_name or Path(cmd[0]).stem) + ".log")
    shown = " ".join(_quote(c) for c in cmd)
    if not quiet:
        ui.show_code("🖥️ 執行指令" + (f"：{title}" if title else ""), shown,
                     note=f"完整輸出會存到 <code>{ui.esc(log_path)}</code>")
    handle = _live_handle()
    lines: list[str] = []
    start = time.time()
    full_env = dict(os.environ)
    full_env.setdefault("PYTHONIOENCODING", "utf-8")
    full_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        full_env.update(env)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=full_env,
                                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                bufsize=1, **_NO_WINDOW)
    except Exception as e:  # noqa: BLE001
        _live_update(handle, [f"無法啟動：{e}"], final=True, ok=False, secs=0)
        return False
    last = 0.0
    with open(log_path, "w", encoding="utf-8") as log:
        assert proc.stdout is not None
        for raw in proc.stdout:
            log.write(raw)
            for piece in raw.rstrip("\n").split("\r"):
                if piece.strip():
                    lines.append(piece[:220])
            now = time.time()
            if not quiet and now - last > 0.25:
                last = now
                _live_update(handle, lines[-tail:], secs=now - start)
            if timeout and now - start > timeout:
                proc.kill()
                lines.append(f"（超過 {timeout:.0f} 秒，已中止）")
                break
        code = proc.wait()
    if not quiet:
        _live_update(handle, lines[-tail:], final=True, ok=(code == 0), secs=time.time() - start)
    return code == 0


def _quote(s: str) -> str:
    return f'"{s}"' if (" " in s or "\n" in s) else s


def _live_handle():
    try:
        from IPython.display import display
        return display({"text/html": _term_html(["（啟動中……）"], False, None, 0), "text/plain": ""},
                       raw=True, display_id=True)
    except Exception:  # noqa: BLE001
        return None


def _term_html(lines, final, ok, secs) -> str:
    body = ui.esc("\n".join(lines)) or " "
    if not final:
        status = f'<span style="color:{ui.C["gold"]}">● 執行中 {secs:.0f} 秒</span>'
    elif ok:
        status = f'<span style="color:{ui.C["mint"]}">✔ 完成（{secs:.1f} 秒）</span>'
    else:
        status = f'<span style="color:{ui.C["red"]}">✘ 指令失敗（{secs:.1f} 秒）</span>'
    return (
        f'<div style="font-family:{ui.MONO};background:#0b0f22;color:#cfd6ff;border:2px solid {ui.C["line"]};'
        f'border-radius:12px;padding:10px 14px;font-size:12.5px;line-height:1.5;max-width:920px;margin:4px 0">'
        f'<div style="font-family:{ui.FONT};font-size:12.5px;margin-bottom:6px">{status}</div>'
        f'<pre style="margin:0;white-space:pre;overflow-x:auto;background:transparent;color:#cfd6ff;'
        f'font-family:{ui.MONO};font-size:12.5px;line-height:1.5;border:0;padding:0 0 4px">{body}</pre></div>'
    )


def _live_update(handle, lines, final=False, ok=None, secs=0.0) -> None:
    if handle is None:
        if final:
            print("\n".join(lines))
        return
    try:
        handle.update({"text/html": _term_html(lines, final, ok, secs), "text/plain": "\n".join(lines)}, raw=True)
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ 下載
def _download(url: str, dest: Path, label: str) -> None:
    """下載檔案（附簡單的進度顯示）。失敗會丟出例外。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "quantize-classroom"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        got, mark = 0, -1
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            got += len(chunk)
            step = int(got / total * 10) if total else got // (10 << 20)
            if step != mark:
                mark = step
                pct = f"{got / total * 100:3.0f}%" if total else ""
                print(f"  ⬇ {label}：{got / 1e6:6.1f} MB {pct}")
    tmp.replace(dest)


def download_model() -> Path:
    """從 Hugging Face 把小史莫（SmolLM2-135M-Instruct）下載到 workspace/models/。"""
    ensure_dirs()
    if (MODEL_DIR / "model.safetensors").exists() and (MODEL_DIR / "config.json").exists():
        print(f"✔ 模型已經在 {MODEL_DIR}，不用重新下載。")
        return MODEL_DIR
    from huggingface_hub import snapshot_download
    print(f"⬇ 正在從 Hugging Face 下載 {MODEL_ID}（約 270 MB）……")
    snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR), allow_patterns=MODEL_FILES)
    return MODEL_DIR


# ------------------------------------------------------------------ llama.cpp 原始碼（轉檔腳本）
_SRC_KEEP = ("convert_hf_to_gguf.py", "conversion/", "gguf-py/", "LICENSE")


def _git(*args) -> bool:
    try:
        return subprocess.run(["git", *args], capture_output=True, stdin=subprocess.DEVNULL, timeout=900,
                              **_NO_WINDOW).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _git_clone_source() -> bool:
    """下載 zip 失敗時的備案：用 git 只抓轉檔需要的那幾個資料夾。"""
    script = SRC_DIR / "convert_hf_to_gguf.py"
    base = ["clone", "--depth", "1", "--branch", LLAMA_TAG]
    shutil.rmtree(SRC_DIR, ignore_errors=True)
    if _git(*base, "--filter=blob:none", "--sparse", LLAMA_REPO, str(SRC_DIR)) and _git(
            "-C", str(SRC_DIR), "sparse-checkout", "set", "--no-cone",
            "/convert_hf_to_gguf.py", "/conversion/", "/gguf-py/", "/LICENSE") and script.exists():
        return True
    shutil.rmtree(SRC_DIR, ignore_errors=True)          # 精簡抓法不行，就整包抓
    if _git(*base, LLAMA_REPO, str(SRC_DIR)) and script.exists():
        return True
    shutil.rmtree(SRC_DIR, ignore_errors=True)
    return False


def get_convert_script() -> Path | None:
    """取得官方轉檔腳本 convert_hf_to_gguf.py（下載 llama.cpp 原始碼，只留需要的部分）。

    有 git 就用 git 只抓需要的資料夾；沒有的話下載壓縮檔再解開。
    都失敗就回傳 None（之後會改用備用轉檔器）。
    """
    ensure_dirs()
    script = SRC_DIR / "convert_hf_to_gguf.py"
    if script.exists():
        return script
    override = os.environ.get("LLAMA_CPP_DIR")
    if override and (Path(override) / "convert_hf_to_gguf.py").exists():
        return Path(override) / "convert_hf_to_gguf.py"
    custom = os.environ.get("QUEST_LLAMA_SRC_URL")
    if not custom and shutil.which("git"):
        print(f"⬇ 用 git 下載 llama.cpp {LLAMA_TAG} 的轉檔腳本（只抓需要的部分，約 4 MB）……")
        if _git_clone_source():
            print(f"✔ 官方轉檔腳本已放在 {SRC_DIR}")
            return script
        print("  git 沒有成功，改成直接下載壓縮檔。")
    url = custom or f"{LLAMA_REPO}/archive/refs/tags/{LLAMA_TAG}.zip"
    zpath = WORK / f"llama.cpp-{LLAMA_TAG}-src.zip"
    try:
        if not zpath.exists():
            print(f"⬇ 下載 llama.cpp {LLAMA_TAG} 原始碼（約 40 MB，裡面有官方轉檔腳本）……")
            _download(url, zpath, "llama.cpp 原始碼")
        n = 0
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
                if not rel or info.is_dir() or not rel.startswith(_SRC_KEEP):
                    continue
                target = SRC_DIR / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
                n += 1
        zpath.unlink(missing_ok=True)
        print(f"✔ 已取出 {n} 個檔案到 {SRC_DIR}")
    except Exception as e:  # noqa: BLE001
        print(f"⚠ 下載 llama.cpp 原始碼失敗：{e}")
        zpath.with_suffix(".zip.part").unlink(missing_ok=True)
    return script if script.exists() else None


# ------------------------------------------------------------------ llama.cpp 執行檔
def _asset_name() -> str | None:
    """這台電腦該下載哪一個預先編譯好的 llama.cpp。"""
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    x64 = machine in ("x86_64", "amd64")
    if system == "Windows" and (arm or x64):
        return f"llama-{LLAMA_TAG}-bin-win-cpu-{'arm64' if arm else 'x64'}.zip"
    if system == "Darwin" and (arm or x64):
        return f"llama-{LLAMA_TAG}-bin-macos-{'arm64' if arm else 'x64'}.tar.gz"
    if system == "Linux" and (arm or x64):
        return f"llama-{LLAMA_TAG}-bin-ubuntu-{'arm64' if arm else 'x64'}.tar.gz"
    return None


def _exe(name: str) -> str:
    return name + (".exe" if platform.system() == "Windows" else "")


def _find_in(folder: Path, name: str) -> Path | None:
    if not folder.exists():
        return None
    for p in sorted(folder.rglob(_exe(name))):
        if p.is_file():
            return p
    return None


def _lib_env(folder: Path) -> dict:
    """讓執行檔找得到放在旁邊的動態函式庫。"""
    key = {"Linux": "LD_LIBRARY_PATH", "Darwin": "DYLD_LIBRARY_PATH"}.get(platform.system(), "PATH")
    old = os.environ.get(key, "")
    return {key: str(folder.resolve()) + (os.pathsep + old if old else "")}


def _works(exe: Path) -> bool:
    try:
        r = subprocess.run([str(exe), "--help"], capture_output=True, timeout=30, stdin=subprocess.DEVNULL,
                           env={**os.environ, **_lib_env(exe.parent)}, **_NO_WINDOW)
        text = (r.stdout + r.stderr).decode("utf-8", "replace").lower()
        return "usage" in text
    except Exception:  # noqa: BLE001
        return False


class LlamaTools:
    """找到的 llama.cpp 執行檔。"""

    def __init__(self, quantize: Path, completion: Path | None, source: str):
        self.quantize = quantize
        self.completion = completion
        self.source = source
        self.env = _lib_env(quantize.parent)

    def __repr__(self) -> str:
        return f"LlamaTools(quantize={self.quantize}, completion={self.completion}, 來源={self.source})"


def _collect(folder: Path, source: str) -> LlamaTools | None:
    q = _find_in(folder, "llama-quantize")
    if q is None or not _works(q):
        return None
    c = _find_in(folder, "llama-completion") or _find_in(folder, "llama-cli")
    return LlamaTools(q, c, source)


def get_llama_tools(verbose: bool = True) -> LlamaTools | None:
    """取得 llama-quantize / llama-completion。

    依序嘗試：
      1. 環境變數 LLAMA_CPP_BIN 指定的資料夾
      2. 之前已經下載到 workspace/llama.cpp-bin 的
      3. 電腦上已經安裝的（PATH 裡找得到）
      4. 從 GitHub 下載官方預先編譯好的版本
    都失敗就回傳 None。
    """
    say = print if verbose else (lambda *a, **k: None)
    override = os.environ.get("LLAMA_CPP_BIN")
    if override:
        t = _collect(Path(override), "LLAMA_CPP_BIN")
        if t:
            return t
    t = _collect(BIN_DIR, "workspace")
    if t:
        say(f"✔ 使用已下載的 llama.cpp：{t.quantize.parent}")
        return t
    on_path = shutil.which("llama-quantize")
    if on_path and _works(Path(on_path)):
        comp = shutil.which("llama-completion") or shutil.which("llama-cli")
        say(f"✔ 使用電腦上已安裝的 llama.cpp：{on_path}")
        return LlamaTools(Path(on_path), Path(comp) if comp else None, "PATH")
    asset = _asset_name()
    if asset is None:
        say(f"⚠ 這台電腦（{platform.system()} {platform.machine()}）沒有官方預先編譯的版本。")
        return None
    url = os.environ.get("QUEST_LLAMA_BIN_URL") or f"{LLAMA_REPO}/releases/download/{LLAMA_TAG}/{asset}"
    archive = WORK / asset
    try:
        ensure_dirs()
        if not archive.exists():
            say(f"⬇ 下載官方預先編譯的 llama.cpp（{asset}，約 15~25 MB）……")
            _download(url, archive, "llama.cpp")
        shutil.rmtree(BIN_DIR, ignore_errors=True)
        BIN_DIR.mkdir(parents=True)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(BIN_DIR)
        else:
            with tarfile.open(archive) as tf:
                try:
                    tf.extractall(BIN_DIR, filter="tar")
                except TypeError:  # 比較舊的 Python 沒有 filter 參數
                    tf.extractall(BIN_DIR)
        archive.unlink(missing_ok=True)
        for p in BIN_DIR.rglob("llama-*"):
            if p.is_file() and not p.is_symlink():
                p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        if platform.system() == "Darwin":  # 解除 macOS 對下載檔案的隔離
            subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(BIN_DIR)], capture_output=True)
    except Exception as e:  # noqa: BLE001
        say(f"⚠ 下載或解壓縮失敗：{e}")
        return None
    t = _collect(BIN_DIR, "官方預編譯版 " + LLAMA_TAG)
    if t is None:
        say("⚠ 下載成功，但執行檔在這台電腦上跑不起來（可能是系統太舊或缺少函式庫）。")
    else:
        say(f"✔ llama.cpp 準備好了：{t.quantize.parent}")
    return t


def install_help() -> list[str]:
    """自動下載失敗時，教同學手動安裝 llama.cpp 的方法。"""
    system = platform.system()
    how = {
        "Windows": "在「終端機」執行 <code>winget install llama.cpp</code>",
        "Darwin": "在「終端機」執行 <code>brew install llama.cpp</code>",
    }.get(system, "執行 <code>brew install llama.cpp</code>，或 <code>conda install -c conda-forge llama.cpp</code>")
    return [
        f"想用官方工具的話可以手動安裝：{how}，裝好後<b>重新啟動 Jupyter</b> 再執行這一格。",
        f"或到 <code>{LLAMA_REPO}/releases</code> 下載對應作業系統的壓縮檔，解壓縮到 <code>{ui.esc(BIN_DIR)}</code>。",
    ]


# ------------------------------------------------------------------ 用 llama.cpp 產生文字
def chat_prompt(question: str, system: str = "You are a helpful assistant.") -> str:
    """把問題包成小史莫看得懂的對話格式（ChatML）。"""
    return (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def llama_generate(tools: LlamaTools, model_path, question: str, max_tokens: int = 80) -> dict | None:
    """用 llama-completion 讓模型回答一個問題。回傳 {'text', 'seconds', 'tokens_per_sec'}。"""
    if tools is None or tools.completion is None:
        return None
    ensure_dirs()
    prompt_file = LOG_DIR / "prompt.txt"
    # llama.cpp 讀檔時會去掉結尾的一個換行，所以多補一個
    prompt_file.write_text(chat_prompt(question) + "\n", encoding="utf-8", newline="\n")
    cmd = [str(tools.completion), "-m", str(model_path), "-f", str(prompt_file), "-n", str(max_tokens),
           "--temp", "0", "--seed", "1", "-no-cnv", "--no-display-prompt", "--simple-io", "-ngl", "0"]
    start = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300, env={**os.environ, **tools.env},
                           stdin=subprocess.DEVNULL, **_NO_WINDOW)
    except Exception:  # noqa: BLE001
        return None
    secs = time.time() - start
    out = r.stdout.decode("utf-8", "replace")
    err = r.stderr.decode("utf-8", "replace")
    text = out.replace("[end of text]", "").replace("<|im_end|>", "").strip()
    if r.returncode != 0 or not text:
        (LOG_DIR / "llama-completion.log").write_text(out + "\n---- stderr ----\n" + err, encoding="utf-8")
        return None
    tps = None
    for line in err.splitlines():           # 例如：eval time = ... (  x ms per token,  123.45 tokens per second)
        if "eval time" in line and "prompt" not in line and "tokens per second" in line:
            try:
                tps = float(line.split("tokens per second")[0].split(",")[-1].strip())
            except ValueError:
                pass
    return {"text": text, "seconds": secs, "tokens_per_sec": tps}


# ------------------------------------------------------------------ 備用轉檔器（不需要 llama.cpp 原始碼）
def read_safetensors(path) -> dict:
    """讀取 .safetensors 檔，回傳 {名稱: float32 的 numpy 陣列}。

    safetensors 的格式很單純：8 bytes 的檔頭長度 + 一段 JSON（每個張量的型別、形狀、位置）+ 原始資料。
    """
    path = Path(path)
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    base = 8 + n
    data = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        raw = data[base + a: base + b]
        dtype = info["dtype"]
        if dtype == "BF16":      # bfloat16 就是 float32 的「前 16 個 bit」，後面補 0 就變回 float32
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif dtype == "F16":
            arr = raw.view(np.float16).astype(np.float32)
        elif dtype == "F32":
            arr = raw.view(np.float32)
        else:
            raise ValueError(f"備用轉檔器不支援 {dtype} 型別的張量：{name}")
        out[name] = arr.reshape(info["shape"])
    return out


def permute_qk(weights: np.ndarray, n_head: int) -> np.ndarray:
    """把 q_proj / k_proj 的列，從 Hugging Face 的排法換成 llama.cpp 的排法。"""
    rows = weights.shape[0]
    return (weights.reshape(n_head, 2, rows // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2).reshape(weights.shape))


def unpermute_qk(weights: np.ndarray, n_head: int) -> np.ndarray:
    """permute_qk 的反操作（把 GGUF 裡的權重裝回 PyTorch 模型時要用）。"""
    rows = weights.shape[0]
    return (weights.reshape(n_head, rows // n_head // 2, 2, *weights.shape[1:])
            .swapaxes(1, 2).reshape(weights.shape))


def backup_convert(model_dir, out_path) -> Path:
    """教授的備用轉檔器：只用 numpy + gguf 套件，把 Hugging Face 的模型轉成 F16 的 GGUF。

    只支援這門課用的模型（Llama 架構 + BPE 詞彙表）。真正的 convert_hf_to_gguf.py
    支援上百種模型，但做的事情本質上一樣：
        1. 把模型的設定（幾層、多寬……）寫進 metadata
        2. 把詞彙表寫進 metadata
        3. 把每個張量改成 GGUF 的命名，存成 F16（一維的小張量保留 F32）
    """
    import gguf

    model_dir, out_path = Path(model_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if cfg.get("model_type") != "llama":
        raise ValueError("備用轉檔器只支援 Llama 架構的模型。")
    tok = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    n_layer = cfg["num_hidden_layers"]
    n_head = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_head)

    w = gguf.GGUFWriter(str(out_path), "llama")
    # 1) 模型設定
    w.add_type("model")
    w.add_name(MODEL_NAME)
    w.add_basename("SmolLM2")
    w.add_size_label("135M")
    w.add_license("apache-2.0")
    w.add_block_count(n_layer)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_rope_freq_base(cfg.get("rope_theta", 10000.0))
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    w.add_vocab_size(cfg["vocab_size"])
    w.add_rope_dimension_count(cfg["hidden_size"] // n_head)

    # 2) 詞彙表
    vocab = tok["model"]["vocab"]
    added = {t["content"]: t for t in tok.get("added_tokens", [])}
    by_id = {i: t for t, i in vocab.items()}
    for t in added.values():
        by_id[t["id"]] = t["content"]
    tokens, types = [], []
    for i in range(cfg["vocab_size"]):
        if i not in by_id:
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
            continue
        text = by_id[i]
        tokens.append(text)
        if text in added:
            special = added[text].get("special") or (text.startswith("<|") and text.endswith("|>"))
            types.append(gguf.TokenType.CONTROL if special else gguf.TokenType.USER_DEFINED)
        else:
            types.append(gguf.TokenType.NORMAL)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("smollm")
    w.add_token_list(tokens)
    w.add_token_types(types)
    special = gguf.SpecialVocab(model_dir, load_merges=True)                    # merges、特殊 token、聊天樣板
    special.add_to_gguf(w, quiet=True)
    if "bos" not in special.add_special_token:
        w.add_add_bos_token(False)
    w.add_add_space_prefix(False)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)

    # 3) 張量
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, n_layer)
    tensors = read_safetensors(model_dir / "model.safetensors")
    for hf_name, arr in tensors.items():
        if hf_name == "lm_head.weight" and cfg.get("tie_word_embeddings", False):
            continue        # 輸出層跟詞嵌入共用同一份權重，不用存兩次
        new = name_map.get_name(hf_name, try_suffixes=(".weight", ".bias"))
        if new is None:
            raise ValueError(f"不認識的張量名稱：{hf_name}")
        # Hugging Face 的 Llama 跟 llama.cpp 排列「旋轉位置編碼」的順序不同，q / k 兩個矩陣的列要換個順序
        if hf_name.endswith("q_proj.weight"):
            arr = permute_qk(arr, n_head)
        elif hf_name.endswith("k_proj.weight"):
            arr = permute_qk(arr, n_kv)
        if arr.ndim >= 2:
            w.add_tensor(new, arr.astype(np.float16))
        else:
            w.add_tensor(new, np.ascontiguousarray(arr, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    return out_path
