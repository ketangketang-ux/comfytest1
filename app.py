@app.function(
    gpu=GPU,
    timeout=86400,
    volumes={DATA: VOL},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.web_endpoint()        # <---- WAJIB AGAR DAPAT URL PUBLIC
@modal.asgi_app()
def launch():
    api = FastAPI()

    # Start comfy in background thread
    def start_comfy():
        ensure_models()
        os.chdir(COMFY)
        run("python3 main.py --listen 0.0.0.0 --port 8188")

    threading.Thread(target=start_comfy, daemon=True).start()

    # Forward ALL requests to ComfyUI backend
    client = httpx.AsyncClient(base_url="http://127.0.0.1:8188")

    @api.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request):
        url = f"http://127.0.0.1:8188/{path}"
        method = request.method

        if method == "GET":
            resp = await client.get(url)
        else:
            body = await request.body()
            resp = await client.post(url, content=body, headers=request.headers)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp.headers
        )

    return api
