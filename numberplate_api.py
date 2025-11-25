# numberplate_api.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms, models
from datetime import datetime
import sqlite3
import io
import os


MODEL_PATH = "/home/yukthi-de/Desktop/TEST CW_2/numberplate/Final_Test /model/APEXAI.pth"
CLASSES_PATH = "/home/yukthi-de/Desktop/TEST CW_2/numberplate/Final_Test /model/classes.txt"
IMG_SIZE = (224, 224)
DB_PATH = "detections.db"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(num_ftrs, len(CLASS_NAMES))
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

infer_tf = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class TEXT,
        confidence REAL,
        detection_time TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()


app = FastAPI(title="Number Plate Classifier API")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Preprocess
        inp = infer_tf(img).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            out = model(inp)
            probs = torch.softmax(out, dim=1)
            conf, pred = probs.max(dim=1)
            conf_val = float(conf.item()) * 100
            pred_class = CLASS_NAMES[int(pred.item())]

        # Current time
        detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO detections (class, confidence, detection_time) VALUES (?,?,?)",
            (pred_class, conf_val, detection_time)
        )
        conn.commit()
        conn.close()

        # Return response
        return JSONResponse({
            "class": pred_class,
            "confidence": conf_val,
            "time": detection_time
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})



@app.get("/logs")
async def get_logs():
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM detections ORDER BY id DESC")
        rows = cur.fetchall()
        conn.close()

        logs = []
        for row in rows:
            logs.append({
                "id": row[0],
                "class": row[1],
                "confidence": row[2],
                "time": row[3]
            })

        return JSONResponse({"logs": logs})

    except Exception as e:
        return JSONResponse({"error": str(e)})

