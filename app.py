import os, shutil, subprocess, modal, zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

DATA_ROOT = "/data/comfy"
DATA_BASE = Path(DATA_ROOT, "ComfyUI")
GPU_TYPE = os.getenv("MODAL_GPU_TYPE", "L4")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "unzip", "ffmpeg",
        "libgl1-mesa-glx", "libglib2.0-0"
    )
    .run_commands(
        "python -m pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "huggingface_hub[hf_transfer]",
        "requests", "tqdm", "insightface", "onnxruntime-gpu"
    )
)

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)
app = modal.App(name="comfyui-2025", image=image)

def run(cmd, cwd=None):
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

def hf_dl(subdir, fn, repo, sf=None):
    t = DATA_BASE / "models" / subdir
    t.mkdir(parents=True, exist_ok=True)
    out = hf_hub_download(repo, fn, subfolder=sf, local_dir="/tmp/dl", local_dir_use_symlinks=False)
    shutil.move(out, t / fn)

@app.function(gpu=GPU_TYPE, timeout=600, volumes={DATA_ROOT: vol})
def setup():
    if not (DATA_BASE / "main.py").exists():
        DATA_BASE.parent.mkdir(parents=True, exist_ok=True)
        run(f"git clone https://github.com/comfyanonymous/ComfyUI {DATA_BASE}")

    run("git pull --ff-only", cwd=DATA_BASE)

    nodes = {
        "ComfyUI-Manager":       "https://github.com/ltdrdata/ComfyUI-Manager.git",
        "comfyui-impact-pack":   "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
        "rgthree-comfy":         "https://github.com/rgthree/rgthree-comfy.git",
        "ComfyUI-ReActor":       "https://github.com/Gourieff/ComfyUI-ReActor.git",
        "ComfyUI-SUPIR":         "https://github.com/cubiq/ComfyUI-SUPIR.git",
        "ComfyUI-InsightFace":   "https://github.com/cubiq/ComfyUI-InsightFace.git",
        "ComfyUI_essentials":    "https://github.com/cubiq/ComfyUI_essentials.git",
        "ComfyUI_IPAdapter_plus":"https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
    }

    for folder, url in nodes.items():
        dst = DATA_BASE / "custom_nodes" / folder
        if dst.exists(): shutil.rmtree(dst)
        run(f"git clone --depth 1 {url} {dst}")

    # insightface buffalo_l
    insight = Path(DATA_ROOT, ".insightface", "models")
    insight.mkdir(parents=True, exist_ok=True)
    if not (insight / "buffalo_l").exists():
        zipf = insight / "buffalo_l.zip"
        run(f"wget -q -O {zipf} https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
        with zipfile.ZipFile(zipf) as z: z.extractall(insight)
        zipf.unlink(missing_ok=True)

    # flux models
    mods = [
        ("checkpoints", "flux1-dev-fp8.safetensors", "camenduru/FLUX.1-dev"),
        ("vae/FLUX",    "ae.safetensors", "comfyanonymous/flux_vae"),
        ("clip/FLUX",   "t5xxl_fp8_e4m3fn.safetensors", "comfyanonymous/flux_text_encoders"),
        ("clip/FLUX",   "clip_l.safetensors", "comfyanonymous/flux_text_encoders"),
    ]
    for sub, fn, repo in mods:
        if not (DATA_BASE / "models" / sub / fn).exists():
            hf_dl(sub, fn, repo)

    vol.commit()
    print("✅ Setup complete")

@app.function(gpu=GPU_TYPE, timeout=86400, volumes={DATA_ROOT: vol})
def launch():
    os.chdir(DATA_BASE)
    print("🔥 Starting ComfyUI @ 0.0.0.0:8188")
    run("python3 main.py --listen 0.0.0.0 --port 8188")
