import base64
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from merged_demo import proctor

app = FastAPI()


class Item(BaseModel):
    name: str
    video_chunk: str
    ai_logs: str
    # price: float
    # tax: Union[float, None] = None


@app.get("/")
def read_root():
    return {"Red Marker": "Proctor AI: Student's worst nightmare"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/vision/")
async def create_item(item: Item):
    open("video.mp4", "wb").write(base64.b64decode(item.video_chunk))
    item.video_chunk = "processed"

    proctor("video.mp4")
    item.ai_logs = open("Monitoring_report.csv", 'r').read()
    return item