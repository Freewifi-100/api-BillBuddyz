import cv2
import Wrap
import Classification
import predictions
from fastapi import HTTPException
import os
from os import getenv
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile
import uuid
import asyncio
import numpy as np

from pydantic import BaseModel
import uvicorn

class Image(BaseModel):
    file: UploadFile = File(...)
    filename : str

# IMAGE_PATH = "./img_post/"

app = FastAPI()

@app.get("/", tags=["Root"])
def read_root():
    return {"Hello": "BillBuddyz!"}

@app.post("/uploadfile/", tags=["Upload"])
async def create_upload_file(file: UploadFile = File(...)):
    # file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    # Save image
    # with open(f"{IMAGE_PATH}{file.filename}", "wb") as f:
    #         f.write(contents)
    # return {"filename": file.filename}
    try:
        # Load image
        # image = cv2.imread(f"{IMAGE_PATH}{file.filename}")
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Classify image
        image_type = Classification.classify_image(image)
        if image_type == 'Handwritten':
            # return {"Type": 'Handwritten'}
            raise HTTPException(status_code=500, detail="Handwritten image")
        else: 
            # Wrap image
            wrapped_image = Wrap.scan(image)  
            predictions_result = predictions.get_predictions(wrapped_image)

            return predictions_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))