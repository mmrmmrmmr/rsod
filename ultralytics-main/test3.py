from ultralytics import YOLO
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# Load a model
# model_y = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', epochs=200, imgsz=640, batch=32)

# model = YOLO('/home/mamingrui/sod/ultralytics-main/runs/detect/yolov8_base/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model
# results = model.val(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', epochs=200, imgsz=640, batch=32)
# result = model.predict(source="/home/mamingrui/sod/visdrone/images/val", device="1", save=True)

# model = YOLO("/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/models/v8/yolov8s_fem.yaml", task='detect')
model = YOLO("/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/models/v8/yolov8s.yaml", task='detect')

import torch
# state = torch.load("/home/mamingrui/sod/ultralytics-main/yolov8n.pt")
# state = torch.load("/home/mamingrui/sod/ultralytics-main/runs/detect/train4/weights/best.pt")

x = torch.rand(1,3,640,640)
# # y = model(x)

results = model.train(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', 
                      epochs=200, 
                      imgsz=640, 
                      batch=16,
                      pretrained=False,
                    #   runs_dir='/home/mamingrui/sod/ultralytics-main/out_dirs',
                      name='nop')
# results = model.val(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', epochs=200, imgsz=640, batch=32)
# result = model.predict(source="/home/mamingrui/sod/visdrone/images/val", device="1", save=True)

xx = model.model(x)
for i in range(3):
    print(xx[i].size())