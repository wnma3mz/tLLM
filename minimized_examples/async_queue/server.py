# main.py

from contextlib import asynccontextmanager
from typing import Any, Dict

from engine import AsyncEngine
from fastapi import BackgroundTasks, FastAPI, HTTPException
from protocol import SequenceRequestData

# 创建队列处理器实例
engine = AsyncEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时开始处理队列
    await engine.start()
    yield
    # 关闭时停止处理队列
    await engine.stop()


app = FastAPI(lifespan=lifespan)


@app.post("/process")
async def process_endpoint(data: SequenceRequestData):
    try:
        # 添加到队列并获取任务ID
        result = await engine.generate(data)
        return result
    except TimeoutError:
        raise HTTPException(status_code=408, detail="Processing timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
