from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model('proj.jpg')
results[0].show()