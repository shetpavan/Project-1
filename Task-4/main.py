from ultralytics import YOLO

model = YOLO("yolov8n.yaml") 

results = model.train(data="google_colab_config.yaml", epochs=50) 