from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from keras.models import load_model
from starlette.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your model
model_path = r"C:\Users\lakshay\Desktop\mri scan\nested_unet_partial.keras"
model = load_model(model_path)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess the image for model prediction."""
    # Resize the image if needed (e.g., 256x256) and scale to [0, 1]
    img_resized = cv2.resize(image, (256, 256))  # Adjust size as per model input
    img_normalized = img_resized / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_normalized, axis=0)  # Add batch dimension

def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """Postprocess the predicted mask to binary (0 and 1)."""
    mask_binary = (mask > 0.5).astype(np.uint8)  # Apply threshold
    return mask_binary

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Read image file
    image = await file.read()
    
    # Convert the image data to numpy array and then to OpenCV format
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"message": "Invalid image."})

    # Preprocess the image
    preprocessed_img = preprocess_image(img)

    # Perform prediction
    predicted_mask = model.predict(preprocessed_img)

    # Postprocess the predicted mask
    postprocessed_mask = postprocess_mask(predicted_mask[0])  # Get first sample

    # Convert mask to a format suitable for returning (e.g., as a list)
    mask_list = postprocessed_mask.flatten().tolist()

    return {"filename": file.filename, "predicted_mask": mask_list}

@app.get("/")
async def root():
    return {"message": "Welcome to the Brain MRI Metastasis Segmentation API!"}
