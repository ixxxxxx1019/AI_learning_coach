"""
统一启动入口 —— 同时启动 FastAPI + Streamlit。

Usage:
    python run.py              # 启动两个服务
    python run.py --api-only   # 仅 FastAPI
    python run.py --ui-only    # 仅 Streamlit

服务端口：
    FastAPI:    http://localhost:8000  (API + Swagger UI /docs)
    Streamlit:  http://localhost:8501  (Web UI)
"""

import argparse
import multiprocessing
import sys


def run_fastapi(port: int = 8000):
    """启动 FastAPI 服务。"""
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=port, log_level="info")


def run_streamlit(port: int = 8501):
    """启动 Streamlit 服务。"""
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            f"--server.port={port}",
            "--server.address=0.0.0.0",
            "--server.headless=true",
            "--browser.gatherUsageStats=false",
        ],
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(description="AI学习教练 启动入口")
    parser.add_argument("--api-only", action="store_true", help="仅启动 FastAPI")
    parser.add_argument("--ui-only", action="store_true", help="仅启动 Streamlit")
    parser.add_argument("--api-port", type=int, default=8000, help="FastAPI 端口")
    parser.add_argument("--ui-port", type=int, default=8501, help="Streamlit 端口")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI学习教练 v0.2.0")
    print("=" * 60)

    if args.api_only:
        print(f"  FastAPI → http://localhost:{args.api_port}")
        print(f"  Swagger → http://localhost:{args.api_port}/docs")
        run_fastapi(args.api_port)
    elif args.ui_only:
        print(f"  Streamlit → http://localhost:{args.ui_port}")
        run_streamlit(args.ui_port)
    else:
        # 同时启动两个服务（多进程，避免 Streamlit 的 event loop 干扰 FastAPI）
        api_proc = multiprocessing.Process(target=run_fastapi, args=(args.api_port,))
        ui_proc = multiprocessing.Process(target=run_streamlit, args=(args.ui_port,))

        print(f"  FastAPI   → http://localhost:{args.api_port}")
        print(f"  Swagger   → http://localhost:{args.api_port}/docs")
        print(f"  Streamlit → http://localhost:{args.ui_port}")
        print("  按 Ctrl+C 停止所有服务")
        print("=" * 60)

        api_proc.start()
        ui_proc.start()

        try:
            api_proc.join()
            ui_proc.join()
        except KeyboardInterrupt:
            api_proc.terminate()
            ui_proc.terminate()
            api_proc.join()
            ui_proc.join()
            print("\n服务已停止。")


if __name__ == "__main__":
    main()
