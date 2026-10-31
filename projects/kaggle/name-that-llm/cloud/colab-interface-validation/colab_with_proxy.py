"""Run the unmodified official Colab CLI with a temporary local-only proxy.

Uses the user's existing Windows proxy on 127.0.0.1:7890. No persistent
proxy/network settings are changed and no network traffic is logged.
"""
import asyncio
import os
import sys

from wsl_proxy_bridge import handle

COLAB = "/home/zhijian/.local/share/codex-colab-cli/bin/colab"

async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    env = dict(os.environ)
    proxy = f"http://127.0.0.1:{port}"
    env.update(HTTPS_PROXY=proxy, HTTP_PROXY=proxy, https_proxy=proxy, http_proxy=proxy,
               NO_PROXY="localhost,127.0.0.1", no_proxy="localhost,127.0.0.1", PYTHONUNBUFFERED="1")
    async with server:
        child = await asyncio.create_subprocess_exec(COLAB, *sys.argv[1:], env=env)
        try:
            return await child.wait()
        finally:
            if child.returncode is None:
                child.terminate()
                await child.wait()

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
