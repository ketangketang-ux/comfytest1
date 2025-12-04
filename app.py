import os
import shutil
import subprocess
from typing import Optional
from huggingface_hub import hf_hub_download
import modal
import threading
import time

# Paths
DATA_ROOT = "/data/comfy"
DATA_BASE = os.path.join(DATA_ROOT, "ComfyUI")
CUSTOM_NODES_DIR = os.path.join(DATA_BASE, "custom_nodes")
MODELS_DIR = os.path.join(DATA_BASE, "models")
TMP_DL = "/tmp/download"
DEFAULT_COMFY_DIR = "/root/comfy/ComfyUI"   # correct name

# Helpers
def git_clone_cmd(node_repo: str, recursive: bool = False, install_reqs: bool = False):
    name = node_repo.split("/")[-1]
    dest = os.path.join(DEFAULT_COMFY_DIR, "custom_nodes", name)
    cmd = f"git clone https://github.com/{node_repo} {dest}"
    if recursive:
        cmd += " --recursive"
    if install_reqs:
        cmd += f" && pip install -r {dest}/requirements.txt"
    return cmd

def hf_download(subdir: str, filename: str, repo_id: str, subfolder: Optional[str] = None):
    out = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        local_dir=TMP_DL
    )
    target = os.path.join(MODELS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    shutil.move(out, os.path.join(target, filename))


# =====================================================================
# BUILD IMAGE
# =====================================================================

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .run_commands([
        "pip install --upgrade pip",
        "pip install --no-cache-dir comfy-cli uv",
        "uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]==0.28.1",
        "comfy --skip-prompt install --nvidia"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Built-in comfy nodes
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

# Git-based nodes (installed in build where allowed)
for repo, flags in [
    ("ssitu/ComfyUI_UltimateSDUpscale", {'recursive': True}),
    ("welltop-cn/ComfyUI-TeaCache", {'install_reqs': True}),
    ("nkchocoai/ComfyUI-SaveImageWithMetaData", {}),
    ("receyuki/comfyui-prompt-reader-node", {'recursive': True, 'install_reqs': True}),
    ("crystian/ComfyUI-Crystools", {'install_reqs': True}),
]:
    image = image.run_commands([git_clone_cmd(repo, **flags)])

# NOTE: InsightFace intentionally NOT installed in build to avoid auth/rate issues.
# It will be installed at runtime (delayed) after ComfyUI boot.

# Runtime model downloads
model_tasks = [
    ("unet/FLUX", "flux1-dev-Q8_0.gguf", "city96/FLUX.1-dev-gguf", None),
    ("clip/FLUX", "t5-v1_1-xxl-encoder-Q8_0.gguf", "city96/t5-v1_1-xxl-encoder-gguf", None),
    ("clip/FLUX", "clip_l.safetensors", "comfyanonymous/flux_text_encoders", None),
    ("checkpoints", "flux1-dev-fp8-all-in-one.safetensors", "camenduru/FLUX.1-dev", None),
    ("loras", "mjV6.safetensors", "strangerzonehf/Flux-Midjourney-Mix2-LoRA", None),
    ("vae/FLUX", "ae.safetensors", "ffxvs/vae-flux", None),
]

# InsightFace ONNX packages (to download via HF)
model_tasks += [
    ("insightface", "det_10g.onnx", "ltdrdata/insightface_models", "antelopev2"),
    ("insightface", "2d106det.onnx", "ltdrdata/insightface_models", "antelopev2"),
    ("insightface", "genderage.onnx", "ltdrdata/insightface_models", "antelopev2"),
    ("insightface", "scrfd_2.5g_bnkps.onnx", "ltdrdata/insightface_models", "antelopev2"),
    ("insightface", "w600k_r50.onnx", "ltdrdata/insightface_models", "antelopev2"),
]

extra_cmds = [
    f"wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth -P {MODELS_DIR}/upscale_models",
]

# =====================================================================
# APP
# =====================================================================

vol = modal.Volume.from_name("comfyui-app", create_if_missing=True)

app = modal.App(name="comfyui", image=image)

@app.function(
    max_containers=1,
    scaledown_window=600,
    timeout=1800,
    gpu=os.environ.get("MODAL_GPU_TYPE", "L4-24GB"),
    volumes={DATA_ROOT: vol},
    secrets=[modal.Secret.from_name("civitai-token")]
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=300)
def ui():

    # --------------------------
    # First-time install / copy
    # --------------------------
    if not os.path.exists(os.path.join(DATA_BASE, "main.py")):
        print("First run detected. Copying ComfyUI to volume...")
        os.makedirs(DATA_ROOT, exist_ok=True)
        if os.path.exists(DEFAULT_COMFY_DIR):
            subprocess.run(f"cp -r {DEFAULT_COMFY_DIR} {DATA_ROOT}/", shell=True, check=True)
        else:
            os.makedirs(DATA_BASE, exist_ok=True)

    # --------------------------
    # Update backend
    # --------------------------
    print("Updating ComfyUI backend...")
    os.chdir(DATA_BASE)
    try:
        r = subprocess.run("git symbolic-ref HEAD", shell=True, capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run("git checkout -B main origin/main", shell=True, check=False)
        subprocess.run("git config pull.ff only", shell=True, check=False)
        subprocess.run("git pull --ff-only", shell=True, check=False)
    except Exception as e:
        print("Backend update error:", e)

    # --------------------------
    # Update / install Manager
    # --------------------------
    manager_dir = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-Manager")
    if os.path.exists(manager_dir):
        try:
            os.chdir(manager_dir)
            subprocess.run("git config pull.ff only", shell=True, check=False)
            subprocess.run("git pull --ff-only", shell=True, check=False)
        except Exception as e:
            print("Manager update error:", e)
    else:
        subprocess.run("comfy node install ComfyUI-Manager", shell=True, check=False)

    # --------------------------
    # pip & comfy-cli upgrade
    # --------------------------
    subprocess.run("pip install --no-cache-dir --upgrade pip", shell=True, check=False)
    subprocess.run("pip install --no-cache-dir --upgrade comfy-cli", shell=True, check=False)

    # Frontend requirements (if present)
    req = os.path.join(DATA_BASE, "requirements.txt")
    if os.path.exists(req):
        subprocess.run(f"/usr/local/bin/python -m pip install -r {req}", shell=True, check=False)

    # --------------------------
    # Manager UI config (user-level)
    # --------------------------
    cfg_dir = os.path.join(DATA_BASE, "user", "default", "ComfyUI-Manager")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "config.ini"), "w") as f:
        f.write(
            "[default]\n"
            "network_mode = private\n"
            "security_level = weak\n"
            "log_to_file = false\n"
            "allow_node_updates = true\n"
            "allow_node_install = true\n"
        )

    # --------------------------
    # BRUTE-FORCE Manager patch (applies to all versions/paths)
    # --------------------------
    try:
        print("Applying brute-force Manager patches...")
        if os.path.exists(manager_dir):
            for root, dirs, files in os.walk(manager_dir):
                for file in files:
                    if not file.endswith(".py"):
                        continue
                    full = os.path.join(root, file)
                    # Force security_level = 'weak' (any variant)
                    subprocess.run(
                        f"sed -i \"s/security_level *= *['\\\"]*[a-zA-Z0-9_]*['\\\"]/security_level = 'weak'/g\" \"{full}\"",
                        shell=True, check=False
                    )
                    # Replace hard-coded 'strict' -> 'weak'
                    subprocess.run(
                        f"sed -i \"s/'strict'/'weak'/g\" \"{full}\"",
                        shell=True, check=False
                    )
                    # Disable verify_signature checks
                    subprocess.run(
                        f"sed -i \"s/verify_signature\\s*(/False and verify_signature(/g\" \"{full}\"",
                        shell=True, check=False
                    )
                    # If verify function used differently, also try to neutralize conditional checks
                    subprocess.run(
                        f"sed -i \"s/if not verify_signature/if False and verify_signature/g\" \"{full}\"",
                        shell=True, check=False
                    )
            print("Manager brute-force patch applied.")
        else:
            print("Manager dir not present for brute-force patch. Skipping.")
    except Exception as e:
        print("Manager brute-force patch error:", e)

    # --------------------------
    # Signature bypass in manager.py (extra safety)
    # --------------------------
    try:
        manager_main = os.path.join(manager_dir, "manager.py")
        if os.path.exists(manager_main):
            subprocess.run(
                f"sed -i 's/if not verify_signature/if False and verify_signature/' \"{manager_main}\"",
                shell=True, check=False
            )
    except Exception as e:
        print("Signature bypass error:", e)

    # --------------------------
    # Make sure directories exist
    # --------------------------
    for d in [CUSTOM_NODES_DIR, MODELS_DIR, TMP_DL, os.path.join(MODELS_DIR, "insightface")]:
        os.makedirs(d, exist_ok=True)

    # --------------------------
    # Download models (HuggingFace)
    # --------------------------
    for sub, fn, repo, subf in model_tasks:
        target = os.path.join(MODELS_DIR, sub, fn)
        if not os.path.exists(target):
            try:
                print(f"Downloading model {fn} into {sub} ...")
                hf_download(sub, fn, repo, subf)
            except Exception as e:
                print("Model download failed:", e)

    # Extra downloads (wget)
    for cmd in extra_cmds:
        try:
            subprocess.run(cmd, shell=True, check=False)
        except Exception as e:
            print("Extra download failed:", e)

    # --------------------------
    # Start ComfyUI
    # --------------------------
    os.environ["COMFY_DIR"] = DATA_BASE
    print("Launching ComfyUI...")
    proc = subprocess.Popen(
        [
            "comfy", "launch", "--",
            "--listen", "0.0.0.0",
            "--port", "8000",
            "--front-end-version", "Comfy-Org/ComfyUI_frontend@latest"
        ],
        cwd=DATA_BASE,
        env=os.environ.copy()
    )

    # --------------------------
    # DELAYED InsightFace install & ensure nodes registered
    # (run in background thread so UI launch isn't blocked)
    # --------------------------
    def install_insightface_postboot():
        # give Comfy some time to initialize internal paths
        time.sleep(8)
        ins = os.path.join(CUSTOM_NODES_DIR, "ComfyUI-InsightFace")
        if not os.path.exists(ins):
            try:
                print("Installing InsightFace node (post-boot)...")
                subprocess.run(
                    f"git clone https://github.com/ltdrdata/ComfyUI-InsightFace {ins}",
                    shell=True, check=False
                )
                req = os.path.join(ins, "requirements.txt")
                if os.path.exists(req):
                    subprocess.run(f"pip install -r {req}", shell=True, check=False)
                print("InsightFace clone + requirements attempted.")
            except Exception as e:
                print("InsightFace install failed:", e)
        else:
            print("InsightFace node already present.")

        # As a safety, touch ComfyUI node folders to encourage reload
        try:
            # trigger a lightweight manager re-scan if manager has a refresh command
            # Try invoking comfy manager CLI if available
            subprocess.run("comfy node rescan || true", shell=True, check=False)
        except Exception:
            pass

        # if needed, try to restart or send SIGHUP to comfy to reload nodes (non-blocking)
        try:
            # graceful restart attempt (non-blocking)
            subprocess.run("pkill -f 'comfy.*launch' || true", shell=True, check=False)
            # start again
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
            print("Attempted a ComfyUI restart to reload nodes.")
        except Exception:
            pass

    thread = threading.Thread(target=install_insightface_postboot, daemon=True)
    thread.start()

    print("ComfyUI should be launching — background tasks started for InsightFace install and manager patches.")
