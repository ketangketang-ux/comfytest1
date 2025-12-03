# modal_comfyui_rework.py
import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download
import modal

# ---------------------------
# Config (ubah sesuai kebutuhan)
# ---------------------------
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"  # kalau lu punya fallback local copy

# Repos yang script akan fallback-clone jika Manager install gagal
FALLBACK_NODE_REPOS = [
    "cubiq/ComfyUI_IPAdapter_plus",
    "rgthree/rgthree-comfy",
    "comfyanonymous/comfyui-impact-pack",
    "nkchocoai/ComfyUI-SaveImageWithMetaData",
    "receyuki/comfyui-prompt-reader-node",
]

# Model tasks (huggingface repo_id, filename, subdir)
MODEL_TASKS = [
    ("checkpoints", "flux1-dev-fp8-all-in-one.safetensors", "camenduru/FLUX.1-dev", None),
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("loras", "mjV6.safetensors", "strangerzonehf/Flux-Midjourney-Mix2-LoRA", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

EXTRA_CMDS = [
    # contoh: download extra weights
    # f"wget -q -O {MODELS_DIR}/upscale_models/RealESRGAN_x4plus_anime_6B.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
]

# ---------------------------
# Helpers
# ---------------------------
def git_clone_to_custom(repo: str):
    name = repo.split("/")[-1]
    dest = os.path.join(CUSTOM_NODES_DIR, name)
    if os.path.exists(dest):
        print(f"[git_clone] already exists: {dest}")
        return
    cmd = f"git clone https://github.com/{repo} {dest}"
    print("[git_clone] running:", cmd)
    subprocess.run(cmd, shell=True, check=False)

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str], hf_token: str):
    os.makedirs(TMP_DL, exist_ok=True)
    print(f"[hf_download] repo={repo_id} filename={filename}")
    try:
        out = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder, local_dir=TMP_DL, token=hf_token)
        target = os.path.join(MODELS_DIR, subdir)
        os.makedirs(target, exist_ok=True)
        shutil.move(out, os.path.join(target, filename))
        print("[hf_download] moved to:", os.path.join(target, filename))
    except Exception as e:
        print("[hf_download] failed:", e)
        raise

# ---------------------------
# Modal image + volume + app
# ---------------------------
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "unzip", "git-lfs", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        # pip + core libs
        "python -m pip install --upgrade pip",
        # comfy-cli helps installing comfy components
        "python -m pip install --no-cache-dir comfy-cli huggingface_hub==0.28.1",
        # insightface + runtime for ONNX (CPU fallback; if you have GPU choose onnxruntime-gpu)
        "python -m pip install --no-cache-dir insightface onnxruntime onnx numpy",
        # common extras many nodes need
        "python -m pip install --no-cache-dir pillow scipy safetensors",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App(name="comfyui-reworked", image=image)

# put secrets names that must exist in your Modal project: hf-token, civitai-token (optional)
@app.function(
    image=image,
    volumes={DATA_ROOT: vol},
    timeout=3600,
    cpu=2,
    memory=8192,
    gpu=os.environ.get("MODAL_GPU_TYPE", None),
    secrets=[modal.Secret.from_name("hf-token"), modal.Secret.from_name("civitai-token")]
)
@modal.web_server(port=8000, startup_timeout=300)
def ui():
    import os
    import subprocess
    hf_token = os.environ.get("hf-token", "")
    print("[ui] hf_token present?:", bool(hf_token))

    # Ensure dirs
    for d in [DATA_ROOT, DATA_BASE, CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)

    # If we have a local fallback comfy, copy once (optional)
    if os.path.exists(DEFAULT_COMFY_DIR) and not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("[ui] copying fallback ComfyUI to volume")
        subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=False)

    # Update repo if it's a git repo (best-effort)
    try:
        if os.path.exists(os.path.join(DATA_BASE, ".git")):
            print("[ui] updating comfy repo (ff-only)")
            subprocess.run("git -C {} config pull.ff only".format(DATA_BASE), shell=True, check=False)
            subprocess.run("git -C {} pull --ff-only".format(DATA_BASE), shell=True, check=False)
    except Exception as e:
        print("[ui] git pull failed:", e)

    # Ensure comfy manager config weak (helps when manager is allowed)
    cfg_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "config.ini")
    with open(cfg_path, "w") as f:
        f.write("[default]\nnetwork_mode = private\nsecurity_level = weak\nlog_to_file = false\n")
    print("[ui] wrote manager config:", cfg_path)

    # Try use comfy node install for common packages (manager-like)
    base_cmd = "comfy node install --yes"
    node_pkgs = [
        "rgthree-comfy",
        "comfyui-impact-pack",
        "comfyui-impact-subpack",
        "comfyui-ipadapter-plus",
        "ComfyUI_Comfyroll_CustomNodes",
        "comfyui-manager-civitai-extension",
    ]
    for pkg in node_pkgs:
        try:
            print("[ui] comfy node install", pkg)
            subprocess.run(f"{base_cmd} {pkg}", shell=True, check=False)
        except Exception as e:
            print("[ui] comfy node install failed for", pkg, e)

    # Fallback: if some nodes not present, clone repos directly to custom_nodes
    print("[ui] ensuring fallback custom nodes...")
    for repo in FALLBACK_NODE_REPOS:
        # check presence by folder name
        name = repo.split("/")[-1]
        dest = os.path.join(CUSTOM_NODES_DIR, name)
        if not os.path.exists(dest):
            try:
                print("[ui] fallback clone:", repo)
                git_clone_to_custom(repo)
            except Exception as e:
                print("[ui] clone failed:", e)

    # Download models using HF token if missing (best-effort)
    print("[ui] downloading model tasks if missing...")
    for sub, fn, repo, subf in MODEL_TASKS:
        try:
            target = os.path.join(MODELS_DIR, sub, fn)
            if not os.path.exists(target):
                print("[ui] hf_download", repo, fn)
                hf_download(sub, fn, repo, subf, hf_token)
        except Exception as e:
            print("[ui] hf download failed for", fn, e)

    # Extra commands (wget etc)
    for cmd in EXTRA_CMDS:
        try:
            print("[ui] running extra cmd:", cmd)
            subprocess.run(cmd, shell=True, check=False)
        except Exception as e:
            print("[ui] extra cmd failed:", e)

    # Start ComfyUI (use comfy CLI)
    os.environ["COMFY_DIR"] = DATA_BASE
    print("[ui] launching comfy (cwd):", DATA_BASE)
    subprocess.Popen(
        ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8000"],
        cwd=DATA_BASE,
        env=os.environ.copy()
    )
    return "ComfyUI starting..."

# End of script
