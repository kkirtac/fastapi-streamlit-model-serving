"""Prediction script. Runs id card visibility prediction with given resnet18 model."""
import sys
sys.path.append('.')
import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile
from modules.model import ResnetLightning
from modules.inference import inference

def read_from_file(file_object):
    arr = np.fromstring(file_object.read(), np.uint8)
    img_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return img_np

# load model
model = ResnetLightning.load_from_checkpoint("/app/models/model_ckpt.ckpt").eval()

app = FastAPI()

@app.post("/predict")
def predict_from_file(file: UploadFile = File(...)):
    """Run prediction of the model with a single image."""

    # repeat the blue channel on green and red channels to receive a 3-channel image
    # because our CNN model requires 3-channel input
    image = read_from_file(file.file)
    image[:, :, 1] = image[:, :, 0]
    image[:, :, 2] = image[:, :, 0]

    # we convert from BGR to RGB because our model is trained so.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    proba_dict = inference(model, image)

    return proba_dict
