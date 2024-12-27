import os
from ultralytics import YOLOv10
config_path = 'config.yaml'

model = YOLOv10("weights/yolov10b.pt")

if __name__ == '__main__':
    results = model.train(data = config_path, epochs = 150, batch = 32, device = 0)