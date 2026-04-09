from fastapi import FastAPI
import torch
from influxdb_client import InfluxDBClient, Point

app = FastAPI()

MODEL_PATH = "/app/model.pth"

model = None  # 🔥 global

# ===== Load model safely =====
def load_model():
    global model
    try:
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
        print("Model loaded OK")
    except Exception as e:
        print("❌ Load model error:", e)

# ===== Startup event =====
@app.on_event("startup")
def startup_event():
    load_model()

# ===== Influx =====
client = InfluxDBClient(
    url="http://influxdb:8086",
    token="my-token",
    org="my-org"
)
write_api = client.write_api()

def log_metric(value):
    try:
        point = Point("prediction").field("value", float(value))
        write_api.write(bucket="my-bucket", record=point)
    except Exception as e:
        print("Influx error:", e)

# ===== API =====
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/predict")
def predict(x: float):

    if model is None:
        return {"error": "model not loaded"}

    x_tensor = torch.tensor([x], dtype=torch.float32)

    with torch.no_grad():
        out = model(x_tensor)

    result = out.item()

    log_metric(result)

    return {"result": result}