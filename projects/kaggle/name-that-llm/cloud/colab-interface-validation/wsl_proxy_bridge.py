"""Temporary WSL-loopback bridge; forwards opaque bytes, never logs traffic."""
import asyncio

WIN_PYTHON = "/mnt/c/Users/34331/AppData/Roaming/uv/tools/kaggle/Scripts/python.exe"
WIN_RELAY = r"C:\计算机学习\深度学习\projects\kaggle\name-that-llm\cloud\colab-interface-validation\windows_proxy_pipe.py"

async def handle(reader, writer):
    process = await asyncio.create_subprocess_exec(
        WIN_PYTHON, WIN_RELAY,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def pipe(source, destination):
        try:
            while data := await source.read(65536):
                destination.write(data)
                await destination.drain()
        except (ConnectionError, BrokenPipeError):
            pass
        finally:
            destination.close()

    tasks = []
    try:
        tasks = [asyncio.create_task(pipe(reader, process.stdin)),
                 asyncio.create_task(pipe(process.stdout, writer))]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        if process.returncode is None:
            process.terminate()
        await process.wait()

async def main():
    server = await asyncio.start_server(handle, "127.0.0.1", 17990)
    print("READY: WSL 127.0.0.1:17990 -> existing Windows loopback proxy", flush=True)
    async with server:
        await server.serve_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
