from ultralytics import YOLO
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import datetime
import yaml
import torch
import sys
sys.path.append('/home/mamingrui/sod/ultralytics-main/SDCluster-main/pretrain/')
from main_p import all_pre

if __name__ == '__main__':

    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    name = 'vis_yolov8s_scam_my3bone2l_3bicross4neck256256256_4fem_my3head_pre'


    model_path_yaml = '/home/mamingrui/sod/ultralytics-main/my_model/'+name+'.yaml'
    out_dirs_path = '/home/mamingrui/sod/ultralytics-main/out_dirs/'+name
    with open(model_path_yaml, 'r', encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)

    if not os.path.exists(out_dirs_path):
        os.makedirs(out_dirs_path)
    
    all_pre(out_dirs_path, yaml_data['pre_path'], model_path_yaml)


    # try:
    #     dir_list = sorted(os.listdir(out_dirs_path), reverse=True)
    #     for i in dir_list:
    #         path = os.path.join(out_dirs_path, i, 'weights')
    #         x = os.listdir(path)
    #         if x != []:
    #             model_path_yaml = os.path.join(path, 'last.pt')
    #             break
    # except:
    #     pass

    # backbone = w
    # x = torch.rand(1,3,640,640)
    model = YOLO(model=model_path_yaml, task='detect')
    # y = model.model(x)[0]
    # backbone = model.model.model[:len(yaml_data['backbone'])]
    # y = backbone(x)
    state_dict = torch.load(out_dirs_path + '/backbone.pt')
    # print(state_dict)
    model.model.model.load_state_dict(state_dict, strict=False)
    # backbone = model.model.model[:len(yaml_data['backbone'])]
    # z = backbone(x)
    # z = model.model(x)[0]
    # xx = y-z

    # deter = yaml_data[]

    def train():
        return model.train(data=yaml_data['data'], 
                            epochs=200, 
                            imgsz=640, 
                            batch=yaml_data['batch'],
                            # batch=32,
                            resume=True,
                            name=now,
                            project=out_dirs_path,
                            close_mosaic=30,
                            deterministic=True,
                            cos_lr=True)
        
    result = train()

    # import torch
    # x = torch.rand(1,3,640,640)
    # xx = model.model(x)
    # for i in range(len(xx)):
    #     print(xx[i].size())