from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import io

app = FastAPI()

# Load the pre-trained VGG16 model
model = tf.keras.models.load_model('model.h5')

# Define class labels
class_labels = ['normal', 'malignant']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Convert binary content to file-like object
        img = image.img_to_array(image.load_img(io.BytesIO(contents), target_size=(150, 150))) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make predictions
        prediction = model.predict(img)
        predicted_class = class_labels[int(prediction[0, 0] > 0.5)]

        return JSONResponse(content={"class": predicted_class, "confidence": float(prediction[0, 0])}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
