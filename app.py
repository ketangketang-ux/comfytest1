###############################
#  COMFYUI SUPER STABLE APP  #
###############################
import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download
import modal

# ==========================================
# CONSTANTS
# ==========================================
DATA_ROOT = "/data"
DATA_BASE = f"{DATA_ROOT}/ComfyUI"
CUSTOM_NODES_DIR = f"{DATA_BASE}/custom_nodes"
MODELS_DIR = f"{DATA_BASE}/models"
TMP_DL = "/tmp/download"
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"


# ==========================================
# HELPERS
# ==========================================
def git_clone(repo: str, recursive=False, install_reqs=False):
    name = repo.split("/")[-1]
    dest = f"{CUSTOM_NODES_DIR}/{name}"

    cmd = f"git clone https://github.com/{repo} {dest}"

    if recursive:
        cmd += " --recursive"

    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"

    return cmd


def hf_get(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    os.makedirs(f"{MODELS_DIR}/{subdir}", exist_ok=True)

    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=TMP_DL
    )

    shutil.move(out, f"{MODELS_DIR}/{subdir}/{filename}")


# ==========================================
#  IMAGE BUILD
# ==========================================
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git", "wget", "libgl1-mesa-glx",
        "libglib2.0-0", "ffmpeg"
    )
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        "comfy --skip-prompt install --nvidia",
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Auto install major custom nodes
image = image.run_commands([
    "comfy node install "
    "rgthree-comfy "
    "comfyui-impact-pack "
    "comfyui-impact-subpack "
    "ComfyUI-YOLO "
    "comfyui-inspire-pack "
    "comfyui_ipadapter_plus "
    "wlsh_nodes "
    "ComfyUI_Comfyroll_CustomNodes "
    "comfyui_essentials "
    "ComfyUI-GGUF "
    "comfyui-manager-civitai-extension"
])

# Additional git-based nodes
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {"recursive": True}),
    ("receyuki/comfyui-prompt-reader-node", {"recursive": True, "install_reqs": True}),
    ("crystian/ComfyUI-Crystools", {"install_reqs": True}),
]:
    image = image.run_commands([git_clone(repo, **flags)])


# ==========================================
# MODEL DOWNLOAD LIST
# ==========================================
model_list = [
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("checkpoints", "flux1-dev-fp8-all-in-one.safetensors", "camenduru/FLUX.1-dev", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
    # IPAdapter FaceID
    ("insightface", "buffalo_l.onnx", "deepghs/insightface", None),
    ("ipadapter", "ip-adapter-faceid-plusv2_sdxl.bin", "h94/IP-Adapter-FaceID", None),
]

extra_downloads = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale",
]


# ==========================================
# APP
# ==========================================
vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)

app = modal.App(
    name="comfyui",
    image=image
)

@app.function(
    gpu="L4",
    timeout=1800,
    volumes={DATA_ROOT: vol},
    secrets=[
        modal.Secret.from_name("civitai-token"),
        modal.Secret.from_name("huggingface-secret"),
    ]
)
@modal.web_server(8000, startup_timeout=300)
def ui():
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    CIVITAI_TOKEN = os.environ.get("civitai-token", "")

    print("HF TOKEN:", bool(HF_TOKEN))
    print("Civitai TOKEN:", bool(CIVITAI_TOKEN))

    # -------------------------------------
    # FIRST INSTALL ComfyUI into volume
    # -------------------------------------
    if not os.path.exists(f"{DATA_BASE}/main.py"):
        print("Copying fresh ComfyUI into volume...")
        os.makedirs(DATA_ROOT, exist_ok=True)
        shutil.copytree(DEFAULT_COMFY_DIR, DATA_BASE, dirs_exist_ok=True)

    # -------------------------------------
    # UPDATE BACKEND
    # -------------------------------------
    os.chdir(DATA_BASE)
    try:
        subprocess.run("git pull --ff-only", shell=True, check=True)
    except:
        print("Git pull failed (safe to ignore)")

    # -------------------------------------
    # UPDATE MANAGER
    # -------------------------------------
    mgr = f"{CUSTOM_NODES_DIR}/ComfyUI-Manager"
    if os.path.exists(mgr):
        os.chdir(mgr)
        subprocess.run("git pull --ff-only", shell=True)

    # -------------------------------------
    # ENABLE WEAK MODE (FIX SECURITY)
    # -------------------------------------
    cfg_dir = f"{DATA_BASE}/user/default/ComfyUI-Manager"
    os.makedirs(cfg_dir, exist_ok=True)

    with open(f"{cfg_dir}/config.ini", "w") as f:
        f.write("[default]\nnetwork_mode = private\nsecurity_level = weak\nlog_to_file = false\n")

    # -------------------------------------
    # PREP DIRECTORIES
    # -------------------------------------
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL]:
        os.makedirs(d, exist_ok=True)

    # -------------------------------------
    # DOWNLOAD MODELS
    # -------------------------------------
    for sub, fn, repo, subf in model_list:
        target = f"{MODELS_DIR}/{sub}/{fn}"
        if not os.path.exists(target):
            try:
                hf_get(sub, fn, repo, subf)
            except Exception as e:
                print("Model DL failed:", fn, e)

    # Extra downloads
    for cmd in extra_downloads:
        subprocess.run(cmd, shell=True)

    # -------------------------------------
    # LAUNCH COMFYUI
    # -------------------------------------
    print("Launching ComfyUI...")
    subprocess.Popen(
        [
            "comfy", "launch", "--",
            "--listen", "0.0.0.0",
            "--port", "8000",
            "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest"
        ],
        cwd=DATA_BASE,
        env=os.environ.copy()
    )

