import sys
sys.path.insert(0, '/home/mamingrui/sod/ultralytics-main/')
from ultralytics import YOLO
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import datetime
import yaml


def get_b(model_path_yaml):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    name = 'vhr_yolov8s_pre'
    name = 'vis_yolov8s_scam_my3bone2l_3bicross4neck256256256_4fem_my3head_pre'

    # name = 'uav_yolov8s_scam_my3bone2l_3bicross4neck256256256_4fem_my3head'


    model_path_yaml = '/home/mamingrui/sod/ultralytics-main/my_model/'+name+'.yaml'
    # out_dirs_path = '/home/mamingrui/sod/ultralytics-main/out_dirs/'+name

    with open(model_path_yaml, 'r', encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
    # backbone = 

    model1 = YOLO(model=model_path_yaml, task='detect')

    backbone1 = model1.model.model[:len(yaml_data['backbone'])]

    model2 = YOLO(model=model_path_yaml, task='detect')

    backbone2 = model2.model.model[:len(yaml_data['backbone'])]
    return backbone1, backbone2

# deter = yaml_data[]

# def train():
#     return model.train(data=yaml_data['data'], 
#                         epochs=200, 
#                         imgsz=640, 
#                         batch=yaml_data['batch'],
#                         # batch=32,
#                         resume=True,
#                         name=now,
#                         project=out_dirs_path,
#                         close_mosaic=30,
#                         deterministic=True,
#                         cos_lr=True)
    
# result = train()

# import torch
# x = torch.rand(1,3,640,640)
# xx = model.model(x)
# for i in range(len(xx)):
#     print(xx[i].size())