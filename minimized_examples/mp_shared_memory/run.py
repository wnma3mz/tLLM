# 项目结构
# project/
# ├── requirements.txt
# ├── config.py
# ├── shared_memory.py  # 上一个回答中的共享内存实现
# ├── engine_process.py # 上一个回答中的引擎实现
# ├── api_server.py     # 上一个回答中的API实现
# └── run.py           # 主启动脚本

import logging
import signal
import subprocess
import sys
import time

import click
from config import Config
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Config.LOG_DIR / "supervisor.log"), logging.StreamHandler()],
)
logger = logging.getLogger("supervisor")


class ProcessManager:
    def __init__(self):
        self.processes = {}
        self.running = True
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}")
        self.shutdown()

    def start_process(self, name, cmd):
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            self.processes[name] = process
            logger.info(f"Started {name} with PID {process.pid}")
            return process
        except Exception as e:
            logger.error(f"Failed to start {name}: {e}")
            return None

    def monitor_process(self, name, process):
        try:
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"{name} exited with code {process.returncode}")
                logger.error(f"stderr: {stderr}")
            return process.returncode
        except Exception as e:
            logger.error(f"Error monitoring {name}: {e}")
            return -1

    def shutdown(self):
        self.running = False
        logger.info("Shutting down all processes...")

        # 首先发送 SIGTERM
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Sending SIGTERM to {name}")
                process.terminate()

        # 等待进程结束
        for i in range(5):  # 最多等待5秒
            if all(process.poll() is not None for process in self.processes.values()):
                break
            time.sleep(1)

        # 如果还有进程没有结束，发送 SIGKILL
        for name, process in self.processes.items():
            if process.poll() is None:
                logger.info(f"Sending SIGKILL to {name}")
                process.kill()


@click.group()
def cli():
    """进程管理CLI"""
    pass


@cli.command()
def start():
    """启动所有服务"""
    Config.init()
    manager = ProcessManager()

    # 启动引擎进程
    for i in range(Config.ENGINE_PROCESS_COUNT):
        manager.start_process(f"engine_{i}", [sys.executable, "engine_process.py"])

    # 启动API服务
    manager.start_process(
        "api",
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api_server:app",
            "--host",
            Config.API_HOST,
            "--port",
            str(Config.API_PORT),
            "--reload",
        ],
    )

    try:
        while manager.running:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        manager.shutdown()


@cli.command()
def status():
    """查看服务状态"""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" in proc.info["name"].lower():
                cmdline = " ".join(proc.info["cmdline"])
                if any(x in cmdline for x in ["engine_process.py", "api_server:app"]):
                    print(f"PID: {proc.info['pid']}, Command: {cmdline}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


@cli.command()
def stop():
    """停止所有服务"""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" in proc.info["name"].lower():
                cmdline = " ".join(proc.info["cmdline"])
                if any(x in cmdline for x in ["engine_process.py", "api_server:app"]):
                    psutil.Process(proc.info["pid"]).terminate()
                    print(f"Terminated process {proc.info['pid']}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


if __name__ == "__main__":
    cli()

# requirements.txt
# fastapi==0.68.0
# uvicorn==0.15.0
# click==8.0.3
# psutil==5.8.0
# numpy==1.21.2
