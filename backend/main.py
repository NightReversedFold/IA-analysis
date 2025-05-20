
from AIhandler import AIhandler
handler: AIhandler = AIhandler()
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse
from pathlib import Path
from utils.retrieve import get_image_details_for_class
import PIL.Image
from tqdm import tqdm

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
        newres = []
        for y in res["results"]:
            newres.append({
                'image_filename': y['image_filename'],
                'bboxes': y["bboxes"]})
        return {"results": newres}
 
    codebookTEXTO = handler.GenerateSeqCodebooks(query)
    filenames = [x["image_filename"] for x in res["results"]]
    TOSORT = []
    for i,file in enumerate(tqdm(filenames)):
        img = PIL.Image.open("data/images/"+file)
        codebookIMAGEN = handler.GenerateImageCodebooks(img)
        dif = handler.FindDifferenceBetweenCodebooks(codebookTEXTO, codebookIMAGEN)
        TOSORT.append({file: dif})
    TOSORT.sort(key=lambda x: list(x.values())[0])
    newres = []
    for y in res["results"]:
        current = [x for x in TOSORT if list(x.keys())[0] == y["image_filename"]][0]
        current = list(current.values())[0]
        newres.append({
            'image_filename': y['image_filename'],
            'bboxes': y["bboxes"],
            'score': current,
        })
    
    newres.sort(key=lambda x: list(x.values())[-1])
    return {"results": newres}
        

@app.get("/images/{image_filename}")
async def read_item(image_filename: str):
    path = Path("data/images/"+image_filename)
    if not path.exists():
        return {"error": "Item not found"}
    
    return FileResponse(path)