from AIhandler import AIhandler
handler: AIhandler = AIhandler()
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, FileResponse # Modified
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
import json
import cv2  # Added
from starlette.background import BackgroundTask  # Added


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
    
    if imag is None: # Added check for None
        return {"error": f"Image {image_filename} not found or could not be processed."}, 404
  
    img_byte_arr = io.BytesIO()
    imag.save(img_byte_arr, format='JPEG') # Save as JPEG
    img_byte_arr.seek(0) # Rewind the buffer to the beginning

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")


@app.post("/video")
async def create_upload_file(file: UploadFile = File(...)):
    temp_input_file_path = None
    temp_output_file_path = None

    try:
        # 1. Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_upload_fileobj:
            shutil.copyfileobj(file.file, tmp_upload_fileobj)
            temp_input_file_path = tmp_upload_fileobj.name
        
        await file.close() # Close the uploaded file stream

        # 2. Get video properties (FPS) from the input video
        cap = cv2.VideoCapture(temp_input_file_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file for reading properties: {temp_input_file_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0: 
            fps = 30.0 

        extracted_frames_rgb = VideoUtility.extract_frames(video_path=temp_input_file_path)
        
        if not extracted_frames_rgb:
            raise ValueError("No frames extracted from video or video is empty.")

        processed_bgr_frames_for_video = []
        
        for frame_rgb in tqdm(extracted_frames_rgb, desc="Processing video frames"):
            yolo_results_list = apocosi(source=frame_rgb, verbose=False) 
            
            if yolo_results_list: 
                yolo_result = yolo_results_list[0] 
                frame_with_boxes_bgr = yolo_result.plot() 
                processed_bgr_frames_for_video.append(frame_with_boxes_bgr)
            else:
                original_frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                processed_bgr_frames_for_video.append(original_frame_bgr)

        if not processed_bgr_frames_for_video:
            raise ValueError("No frames were processed or available for video creation.")

        h_out, w_out, _ = processed_bgr_frames_for_video[0].shape
        
        _temp_output_file_obj = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output_file_path = _temp_output_file_obj.name
        _temp_output_file_obj.close() 

        # Try common FourCC codes
        # For MP4, 'mp4v' is common on many systems, 'avc1' for H.264, or 'h264'
        # For AVI, 'XVID' or 'MJPG' are common
        # The availability depends on the OpenCV build and installed codecs.
        # Using integer 0x7634706d for 'mp4v' as a fallback if string version fails
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        except AttributeError:
            print("cv2.VideoWriter_fourcc not found, trying direct integer for 'mp4v'")
            fourcc = 0x7634706d # Integer for 'mp4v'
        
        out_video_writer = cv2.VideoWriter(temp_output_file_path, fourcc, float(fps), (w_out, h_out))
        
        if not out_video_writer.isOpened():
            print(f"Failed to open VideoWriter with mp4v (fourcc: {fourcc}). Trying XVID.")
            try:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            except AttributeError:
                print("cv2.VideoWriter_fourcc not found, trying direct integer for 'XVID'")
                fourcc = 0x44495658 # Integer for 'XVID'
            out_video_writer = cv2.VideoWriter(temp_output_file_path, fourcc, float(fps), (w_out, h_out))
            if not out_video_writer.isOpened():
                 raise RuntimeError(f"Could not open VideoWriter for the output file: {temp_output_file_path}. Tried mp4v and XVID.")

        for bgr_frame in processed_bgr_frames_for_video:
            out_video_writer.write(bgr_frame)
        
        out_video_writer.release()
        
        return FileResponse(
            path=temp_output_file_path, 
            media_type="video/mp4", 
            filename="processed_video.mp4",
            background=BackgroundTask(os.remove, temp_output_file_path)
        )

    except Exception as e:
        print(f"Error processing video: {e}") 
        if temp_output_file_path and os.path.exists(temp_output_file_path):
            try:
                os.remove(temp_output_file_path)
            except Exception as cleanup_exc:
                print(f"Error during cleanup of output file: {cleanup_exc}")
        return {"error": f"An error occurred: {str(e)}"}
    finally:
        if temp_input_file_path and os.path.exists(temp_input_file_path):
            try:
                os.remove(temp_input_file_path)
            except Exception as cleanup_exc:
                print(f"Error during cleanup of input file: {cleanup_exc}")