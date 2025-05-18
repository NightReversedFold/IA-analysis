
from AIhandler import AIhandler
handler = AIhandler()
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from pathlib import Path
from utils.retrieve import get_image_details_for_class
import PIL.Image

app = FastAPI()

class LabelQuery(BaseModel):
    classes: list
    queryFORAI: str | None = None
    inclusivo: bool = True

@app.post("/query/")
async def queryAI(label_query: LabelQuery):
    query, classes = label_query.queryFORAI, label_query.classes
    res = {"results": get_image_details_for_class(classes, Path("data/Annotations"), label_query.inclusivo)}
    if len(res) == 0:
        return {"error": "No results found"}
    if query is None:
        return res
 
    codebookTEXTO = handler.GenerateSeqCodebooks(query)
    filenames = [x["image_filename"] for x in res["results"]]
    TOSORT = []
    for file in filenames:
        img = PIL.Image.open("data/images/"+file)
        codebookIMAGEN = handler.GenerateImageCodebooks(img)
        dif = handler.FindDifferenceBetweenCodebooks(codebookTEXTO, codebookIMAGEN)
        TOSORT.append({file: dif})
    TOSORT.sort(key=lambda x: list(x.values())[0])
    return {TOSORT, res}
        

@app.get("/images/{image_filename}")
async def read_item(image_filename: str):
    path = Path("data/images/"+image_filename)
    if not path.exists():
        return {"error": "Item not found"}
    
    return FileResponse(path)