import os
import subprocess
import shutil
from huggingface_hub import hf_hub_download
import modal

# --- CONFIG: gunakan path yang kecil & unik supaya tidak bentrok dengan image ---
DATA_ROOT = "/comfy_data"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"  # optional copy source if you pre-bake one in image

# --- Modal volume (nama volume tetap) ---
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)

# --- Build image (sesuaikan) ---
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli huggingface_hub"
    ])
    # jangan buat DATA_ROOT di image build (biar mount bisa terjadi)
)

app = modal.App("comfyui", image=image)

# --- IMPORTANT: reference the modal secret by its name (huggingface-secret) ---
@app.function(
    image=image,
    volumes={DATA_ROOT: vol},   # mount volume to /comfy_data (must NOT already exist+have content in image)
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],  # <-- use secret name here
)
def ui():
    import os, subprocess
    print("[ui] start")

    # read HF token from environment
    hf_token = os.environ.get("HF_TOKEN")
    print("[ui] HF_TOKEN present?:", bool(hf_token))

    # ensure dirs
    os.makedirs(DATA_BASE, exist_ok=True)
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(TMP_DL, exist_ok=True)

    # simple hf download example (if need)
    # hf_hub_download will use HF_TOKEN in env automatically
    try:
        # contoh: download det_10g.onnx dari repo deepghs/insightface
        # file must exist in repo and be accessible by token/privacy rules
        out = hf_hub_download(repo_id="deepghs/insightface", filename="det_10g.onnx", local_dir=TMP_DL)
        shutil.move(out, os.path.join(MODELS_DIR, "insightface", "det_10g.onnx"))
        print("Downloaded insightface det_10g.onnx")
    except Exception as e:
        print("hf download failed (ok if already present or token missing):", e)

    # launch comfy if present
    comfy_main = os.path.join(DATA_BASE, "main.py")
    if os.path.exists(comfy_main):
        env = os.environ.copy()
        env["COMFY_DIR"] = DATA_BASE
        subprocess.Popen(["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000"], cwd=DATA_BASE, env=env)
        print("ComfyUI launched")
    else:
        print("ComfyUI not found at", DATA_BASE)

    return "ok"
