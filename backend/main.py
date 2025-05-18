
from AIhandler import AIhandler
handler = AIhandler()
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
import PIL
from pathlib import Path

app = FastAPI()

class LabelQuery(BaseModel):
    classes: list
    queryFORAI: str

@app.post("/query/")
async def queryAI(label_query: LabelQuery):
    print(label_query)

@app.get("/{image_id}")
async def read_item(image_filename: str):
    path = Path("data/images/"+image_filename+".jpg")
    if not path.exists():
        return {"error": "Item not found"}
    image = PIL.Image.open(path)
    return FileResponse(image)