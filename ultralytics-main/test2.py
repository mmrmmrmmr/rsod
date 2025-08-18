from ultralytics import YOLO
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# Load a model
# model_y = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', epochs=200, imgsz=640, batch=32)

# model = YOLO('/home/mamingrui/sod/ultralytics-main/runs/detect/train6/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model
# results = model.val(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', epochs=200, imgsz=640, batch=32)
# result = model.predict(source="/home/mamingrui/sod/visdrone/images/val", device="1", save=True)

model = YOLO("/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/models/v8/yolov8_try2.yaml", task='detect')
# # model._load("/home/mamingrui/sod/ultralytics-main/yolov8n.pt")
# model_s_o = model.state_dict()
import torch
state = torch.load("/home/mamingrui/sod/ultralytics-main/yolov8n.pt")
# state = torch.load("/home/mamingrui/sod/ultralytics-main/runs/detect/train4/weights/best.pt")

x = torch.rand(1,3,640,640)
# y = model(x)
# model.model.model.load_state_dict(state['model'].model[0:10].state_dict(),strict=False)
# model_s_a = model.state_dict()

# results = model.train(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', epochs=100, imgsz=640, batch=16)
# # results = model.val(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', epochs=200, imgsz=640, batch=32)
# result = model.predict(source="/home/mamingrui/sod/visdrone/images/val", device="1", save=True)

