
from AIhandler import AIhandler
handler: AIhandler = AIhandler()
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from pathlib import Path
from utils.retrieve import get_image_details_for_class, parse_voc_annotation
from utils.preproccess import utility
import PIL.Image
from tqdm import tqdm
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import PIL.ImageDraw
import io
from ultralytics import YOLO
import tempfile
import shutil
import os
import json # Add this import


apocosi = YOLO("yolo11n.pt")
VideoUtility = utility()
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
    
  
    img_byte_arr = io.BytesIO()
    imag.save(img_byte_arr, format='JPEG') # Save as JPEG
    img_byte_arr.seek(0) # Rewind the buffer to the beginning

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")







@app.post("/video")
async def create_upload_file(file: UploadFile=File(...)):
    tmp_video_path = None 
    try:
        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_video_path = tmp.name
        
        print(f"Temporary video file saved at: {tmp_video_path}")
        
        frames = VideoUtility.extract_frames(video_path=tmp_video_path)
        
        res = []
        video =[]
        for frame in frames:
            img = PIL.Image.fromarray(frame)
            yolo_results_list = apocosi(source=img) # Returns a list of Results objects
            if yolo_results_list:
               
                json_string_output = yolo_results_list[0].to_json()
                res.append(json.loads(json_string_output))
            else:
                res.append([]) # Handle cases where no objects are detected in a frame
            draw = PIL.ImageDraw.Draw(img)
            bboxes = []
            if yolo_results_list:
                for result in yolo_results_list:
                    # Extract bounding boxes from the results
                    bboxes.extend(result.boxes.xyxy.tolist())
                for bbox in bboxes:
                    xmin, ymin, xmax, ymax = bbox
                    # Ensure coordinates are integers for drawing
                    draw.rectangle([int(xmin), int(ymin), int(xmax, int(ymax))], outline="red", width=3)
            video.append(np.array(img))

        return {"data": res}
    finally:
        # Ensure the temporary file is deleted after processing
        if tmp_video_path and os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)
        # Close the uploaded file
        await file.close()