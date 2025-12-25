import traceback
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


# Forward OpenAI requests to a model instance
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Currently, we do not support streaming chat completions
    try:
        request_data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    forward_headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in ["host", "content-length", "transfer-encoding"]
    }
    # for experience data recording, we need to return token ids and logprobs
    request_data["return_token_ids"] = True
    request_data["logprobs"] = True
    # temperature must be set from config, ignore user's input
    request_data["temperature"] = request.app.state.temperature
    url, model_version = await request.app.state.service.allocate_model()
    try:
        async with httpx.AsyncClient(timeout=request.app.state.inference_timeout) as client:
            resp = await client.post(
                f"{url}/v1/chat/completions", json=request_data, headers=forward_headers
            )
    except Exception:
        return Response(
            status_code=500,
            content=f"Error forwarding request to model at {url}: {traceback.format_exc()}",
        )
    resp_data = resp.json()
    await request.app.state.service.record_experience(resp_data, model_version)
    return JSONResponse(content=resp_data)


@app.get("/v1/models")
async def show_available_models(request: Request):
    if hasattr(request.app.state, "models"):
        return JSONResponse(content=request.app.state.models)
    url, _ = await request.app.state.service.allocate_model(increase_count=False)
    async with httpx.AsyncClient() as client:
        print(f"Fetching models from {url}/v1/models")
        resp = await client.get(f"{url}/v1/models")
    request.app.state.models = resp.json()
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


@app.post("/feedback")
async def feedback(request: Request):
    """Receive feedback for the current session."""
    body = await request.json()
    reward = body.get("reward")
    msg_ids = body.get("msg_ids")
    task_id = body.get("task_id")
    run_id = body.get("run_id", 0)
    if msg_ids is None or reward is None:
        return JSONResponse(status_code=400, content={"error": "msg_ids and reward are required"})
    if not isinstance(msg_ids, list) or not isinstance(reward, (int, float)):
        return JSONResponse(
            status_code=400, content={"error": "msg_ids must be a list and reward must be a number"}
        )
    await request.app.state.service.record_feedback(
        reward=reward, msg_ids=msg_ids, task_id=task_id, run_id=run_id
    )
    return JSONResponse(content={"status": "success"})


@app.post("/commit")
async def commit(request: Request):
    """Commit the current experiences."""
    await request.app.state.service.submit_experiences()
    return JSONResponse(content={"status": "success"})


async def serve_http(app: FastAPI, host: str, port: int) -> None:
    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)
    await server.serve()


async def run_app(service, listen_address: str, port: int) -> None:
    app.state.service = service
    app.state.temperature = service.explorer.config.model.temperature
    app.state.inference_timeout = service.explorer.config.synchronizer.sync_timeout
    await serve_http(app, listen_address, port)
