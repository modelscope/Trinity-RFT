import traceback

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()


# Forward openAI requests to a model instance


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Currently, we do not support streaming chat completions
    body = await request.json()
    if "return_token_ids" not in body:
        body["return_token_ids"] = True
    url = await request.app.state.service.allocate_model()
    try:
        async with httpx.AsyncClient(timeout=request.app.state.inference_timeout) as client:
            resp = await client.post(f"{url}/v1/chat/completions", json=body)
    except Exception:
        return Response(
            status_code=500,
            content=f"Error forwarding request to model at {url}: {traceback.format_exc()}",
        )
    resp_data = resp.json()
    await request.app.state.service.record_experience(resp_data)
    return JSONResponse(content=resp_data)


@app.get("/v1/models")
async def show_available_models(request: Request):
    body = await request.json()
    url = await request.app.state.service.allocate_model(increase_count=False)
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{url}/v1/models", json=body)
    return JSONResponse(content=resp.json())


@app.get("/health")
async def health(request: Request) -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/metrics")
async def metrics(request: Request):
    """Get the metrics of the service."""
    metrics = request.app.state.service.collect_metrics()
    metrics["explore_step_num"] = request.app.state.service.explorer.explore_step_num
    return JSONResponse(content=metrics)


async def serve_http(app: FastAPI, host: str, port: int = None):
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


async def run_app(service, listen_address: str, port: int = None) -> FastAPI:
    app.state.service = service
    app.state.inference_timeout = service.explorer.config.synchronizer.sync_timeout
    print(f"API server running on {listen_address}:{port}")
    await serve_http(app, listen_address, port)
