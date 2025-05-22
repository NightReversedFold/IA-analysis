
from AIhandler import AIhandler
handler: AIhandler = AIhandler()
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from pathlib import Path
from utils.retrieve import get_image_details_for_class, parse_voc_annotation
import PIL.Image
from tqdm import tqdm
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import PIL.ImageDraw
import io

app = FastAPI()

origins = [
"http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

def load_image_and_draw_bboxes(image_path: Path, bboxes: list[list[int]]) -> PIL.Image.Image | None: # Corrected type hint
    
    try:
        image = PIL.Image.open(image_path).convert("RGB")
        draw = PIL.ImageDraw.Draw(image)
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            # Ensure coordinates are integers for drawing
            draw.rectangle([int(xmin), int(ymin), int(xmax), int(ymax)], outline="red", width=3)
        # return PIL.Image.fromarray(np.array(image)) # This returns a PIL.Image object
        return image # Return the modified image directly
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading or drawing on image {image_path}: {e}")
        return None   

@app.get("/images/{image_filename}")
async def read_item(image_filename: str):
    path = Path("data/images/"+image_filename)
    
    pathToXML = Path("data/Annotations/"+image_filename.replace(".jpg", ".xml"))
    
    annotation_data = parse_voc_annotation(pathToXML)
    specific_bboxes_for_class = []
    if annotation_data and annotation_data.get('objects'): # Check for objects
        for obj in annotation_data.get('objects', []):        
            specific_bboxes_for_class.append(obj['bbox'])
    
    imag: PIL.Image.Image | None = load_image_and_draw_bboxes(path, specific_bboxes_for_class)
    
    # Save PIL image to a BytesIO buffer
    img_byte_arr = io.BytesIO()
    imag.save(img_byte_arr, format='JPEG') # Save as JPEG
    img_byte_arr.seek(0) # Rewind the buffer to the beginning

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")