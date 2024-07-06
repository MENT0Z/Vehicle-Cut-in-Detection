from ultralytics import YOLO
model = YOLO("model1.pt")
vdo = "videos/carss.mp4"
result = model.predict(vdo,show=True)
print(result)