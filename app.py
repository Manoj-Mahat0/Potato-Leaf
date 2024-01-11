

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("potatoes.h5")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Welcome to Beeown"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the file as an image
        image = read_file_as_image(await file.read())

        # Resize the image to match the expected input shape of the model (256x256)
        resized_image = tf.image.resize(image, (256, 256))

        # Expand dimensions to create a batch of size 1
        img_batch = np.expand_dims(resized_image, 0)

        # Make predictions
        predictions = MODEL.predict(img_batch)

        # Get the predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8000, debug=True)

