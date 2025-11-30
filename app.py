import os
import requests
import modal

app = modal.App("comfyui-simple")

VOL = modal.Volume.from_name("comfy-vol", create_if_missing=True)
MODEL_DIR = "/data/comfy/ComfyUI/models/checkpoints"

@app.function(
    timeout=600,
    volumes={"/data": VOL},
    secrets=[modal.Secret.from_name("civitai-token")]  # <<< INI NAMA SECRET
)
def download_basemodel():

    os.makedirs(MODEL_DIR, exist_ok=True)

    token = os.environ.get("CIVITAI_TOKEN")  # <<< KEY SECRET
    if not token:
        raise Exception("❌ CIVITAI_TOKEN tidak ditemukan di Modal Secret 'civitai-token'")

    # URL base model
    url = "https://civitai.com/api/download/models/2285644?type=Model&format=SafeTensor&size=pruned&fp=fp16"

    dst = f"{MODEL_DIR}/basemodel_fp16.safetensors"

    print(f"⬇️ Downloading Civitai base model ke {dst} ...")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    with requests.get(url, headers=headers, stream=True) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

    print("🎉 DONE! Base model tersimpan di volume.")
