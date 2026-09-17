"""Wrapper to launch vllm.entrypoints.openai.api_server with numpy compat patch."""
import numpy as np
np.long = np.int64
np.ulong = np.uint64

import sys
from vllm.entrypoints.openai.api_server import run_server

if __name__ == "__main__":
    import asyncio
    from vllm.entrypoints.openai.api_server import build_async_engine_client, build_app, setup_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.api_server import FlexibleArgumentParser, make_arg_parser
    
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args()
    asyncio.run(run_server(args))
