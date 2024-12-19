from fastapi import FastAPI, File, UploadFile, Form
import shutil
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import shutil
from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer
import pickle
import cv2
from tensorflow.keras.models import load_model
import numpy as np


app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIRECTORY = "uploads"

# Create the directory if it doesn't exist
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

@app.post("/upload/")
async def upload_audio(file: UploadFile = File(...), description: str = Form(...)):
    # Path to save the file
    file_location = f"{UPLOAD_DIRECTORY}/{file.filename}"
    
    # Save the file to the specified directory
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "info": f"File '{file.filename}' has been saved.",
        "description": description  # Return the string description
    }

@app.post("/message/")
async def post_message(message: str = Form(...)):
    return {
        "message": f"Received message: {message}"
    }