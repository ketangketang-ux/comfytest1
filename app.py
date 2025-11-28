# ==========================
# comfyui_appc.py  (FULL, Modal ≥ 0.63)
# ==========================
import os, shutil, subprocess, modal
from pathlib import Path
from huggingface_hub import hf_hub_download

DATA_ROOT = "/data/comfy"
DATA_BASE = Path(DATA_ROOT, "ComfyUI")
GPU_TYPE  = os.getenv("MODAL_GPU_TYPE", "L4")

# ---------- IMAGE ----------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "unzip", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .run_commands(
        "python -m pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "insightface",
        "onnxruntime-gpu",
        "requests",
        "tqdm",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="comfyui-2025", image=image)

# ---------- helpers ----------
def run(cmd: str, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def hf_dl(subdir: str, filename: str, repo_id: str, subfolder: str | None = None):
    target = DATA_BASE / "models" / subdir
    target.mkdir(parents=True, exist_ok=True)
    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir="/tmp/dl",
        local_dir_use_symlinks=False,
    )
    shutil.move(out, target / filename)

# ---------- MAIN ----------
@app.function(
    gpu=GPU_TYPE,
    timeout=3600,
    volumes={DATA_ROOT: vol},
    scaledown_window=300,
    max_containers=1,
)
@modal.web_server(8188)
def ui():
    # 1. clone / update ComfyUI core
    if not (DATA_BASE / "main.py").exists():
        DATA_BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {DATA_BASE}")
    os.chdir(DATA_BASE)
    run("git config pull.ff only && git pull --ff-only")

    # 2. Manager
    mgr = DATA_BASE / "custom_nodes" / "ComfyUI-Manager"
    if mgr.exists():
        os.chdir(mgr); run("git pull --ff-only"); os.chdir(DATA_BASE)
    else:
        run("git clone https://github.com/ltdrdata/ComfyUI-Manager.git custom_nodes/ComfyUI-Manager")

    # 3. Node hits 2025 – inside ui() → no UnboundLocalError
    repos = [
        ("rgthree-comfy",          "https://github.com/rgthree/rgthree-comfy.git"),
        ("comfyui-impact-pack",    "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"),
        ("ComfyUI-ReActor",        "https://github.com/Gourieff/ComfyUI-ReActor.git"),
        ("ComfyUI-SUPIR",          "https://github.com/cubiq/ComfyUI-SUPIR.git"),
        ("ComfyUI-InsightFace",    "https://github.com/cubiq/ComfyUI-InsightFace.git"),
        ("ComfyUI_essentials",     "https://github.com/cubiq/ComfyUI_essentials.git"),
        ("comfyui-ipadapter-plus", "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"),
        ("ComfyUI-KJNodes",        "https://github.com/kijai/ComfyUI-KJNodes.git"),
        ("ComfyUI_Mira",           "https://github.com/Mira-Geoscience/ComfyUI_Mira.git"),
    ]
    for folder, url in repos:
        dst = DATA_BASE / "custom_nodes" / folder
        if dst.exists():
            shutil.rmtree(dst)
        try:
            run(f"git clone --depth 1 {url} {dst}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Skip {folder} – clone failed")
            continue

               # 4. InsightFace model – GitHub mirror + native extract
    import zipfile, io
    insight_vol = Path(DATA_ROOT, ".insightface", "models")
    insight_vol.mkdir(parents=True, exist_ok=True)
    target_dir = insight_vol / "buffalo_l"

    if not target_dir.exists():
        print("⬇️  Download InsightFace buffalo_l (GitHub mirror)...")
        zip_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
        zip_path = insight_vol / "buffalo_l.zip"

        # download
        run(f"wget -q --show-progress -O {zip_path} {zip_url}")
        # ekstrak pakai Python (no unzip binary)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(insight_vol)
        zip_path.unlink(missing_ok=True)
        print("✅ buffalo_l extracted")
    # 5. Model dasar (FLUX + VAE + CLIP)
    mods = [
        ("checkpoints", "flux1-dev-fp8.safetensors", "camenduru/FLUX.1-dev", None),
        ("vae/FLUX", "ae.safetensors", "comfyanonymous/flux_vae", None),
        ("clip/FLUX", "t5xxl_fp8_e4m3fn.safetensors", "comfyanonymous/flux_text_encoders", None),
        ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ]
    for sub, fn, repo, sf in mods:
        if not (DATA_BASE / "models" / sub / fn).exists():
            hf_dl(sub, fn, repo, sf)

    # 6. persist & run
vol.commit()
print("🚀 ComfyUI ready – Modal will start main.py")
# jangan panggil main.py lagi, serahkan ke @modal.web_server
import time
while True:
    time.sleep(3600)
