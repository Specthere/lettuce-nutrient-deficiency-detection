
from ultralytics import YOLO

# Memuat pretrained YOLOv12 Nano
model = YOLO("yolo12n.pt")

print("Memulai training YOLOv12...")

# Proses training
results = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    patience=20,
    workers=4,
    project="runs/detect",
    name="lettuce_yolo12",
    plots=True,
    save=True
)

print("\nTraining selesai!")
print("Model terbaik:")
print("runs/detect/lettuce_yolo12/weights/best.pt")
