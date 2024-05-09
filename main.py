import cv2
import Wrap
import Classification
import predictions
from fastapi import HTTPException
import os
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile
import uuid
import asyncio

from pydantic import BaseModel
import uvicorn

class Image(BaseModel):
    file: UploadFile = File(...)
    filename : str

IMAGE_PATH = "./img_post/"

app = FastAPI()

@app.get("/", tags=["Root"])
def read_root():
    return {"Hello": "World"}

@app.post("/uploadfile/", tags=["Upload"])
async def create_upload_file(file: UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()
    # Save image
    with open(f"{IMAGE_PATH}{file.filename}", "wb") as f:
            f.write(contents)
    # return {"filename": file.filename}
    try:
        # Load image
        image = cv2.imread(f"{IMAGE_PATH}{file.filename}")
        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Classify image
        image_type = Classification.classify_image(image)
        if image_type == 'Handwritten':
            os.remove(f"{IMAGE_PATH}{file.filename}")
            return {"Type": 'Handwritten'}
        else: 
            # Wrap image
            wrapped_image = Wrap.add_performance(image)  
            predictions_result = predictions.getPredictions(wrapped_image)
            
            # Delete image by path if needed
            os.remove(f"{IMAGE_PATH}{file.filename}")

            return predictions_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def main():
    config = uvicorn.Config("main:app", port=5000, log_level="info", reload=True)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())