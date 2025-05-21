"""OpenAI API server related tools.

Modified from vllm/entrypoints/openai/api_server.py
"""
import asyncio
import functools

from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app,
    create_server_socket,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser, set_ulimit


async def run_server_in_ray(args, engine_client):
    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host, args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()
    app = build_app(args)

    vllm_config = await engine_client.get_vllm_config()
    await init_app_state(engine_client, vllm_config, app.state, args)

    await patch_and_serve_http(app, sock, args)

    # # NB: Await server shutdown only after the backend context is exited
    # try:
    #     await shutdown_task
    # finally:
    #     sock.close()


def dummy_add_signal_handler(self, *args, **kwargs):
    # DO NOTHING HERE
    pass


async def patch_and_serve_http(app, sock, args):
    """Patch the add_signal_handler method and serve the app."""
    loop = asyncio.get_event_loop()
    original_add_signal_handler = loop.add_signal_handler
    loop.add_signal_handler = functools.partial(dummy_add_signal_handler, loop)

    try:
        shutdown_task = await serve_http(
            app,
            sock=sock,
            enable_ssl_refresh=args.enable_ssl_refresh,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True,
            timeout_keep_alive=10,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
        )
        await shutdown_task
    finally:
        loop.add_signal_handler = original_add_signal_handler
        sock.close()


async def run_api_server_in_ray_actor(async_llm, host: str, port: int, model_path: str):
    parser = FlexibleArgumentParser(description="Run the OpenAI API server.")
    args = make_arg_parser(parser)
    args = parser.parse_args(["--host", str(host), "--port", str(port), "--model", model_path])
    print(args)
    await run_server_in_ray(args, async_llm)
