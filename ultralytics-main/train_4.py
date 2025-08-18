from ultralytics import YOLO
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import datetime
import yaml

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

name = 'dai_yolov8m'

model_path_yaml = '/home/mamingrui/sod/ultralytics-main/my_model/'+name+'.yaml'
out_dirs_path = '/home/mamingrui/sod/ultralytics-main/out_dirs/'+name

with open(model_path_yaml, 'r', encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

try:
    dir_list = sorted(os.listdir(out_dirs_path), reverse=True)
    for i in dir_list:
        path = os.path.join(out_dirs_path, i, 'weights')
        x = os.listdir(path)
        if x != []:
            model_path_yaml = os.path.join(path, 'last.pt')
            break
except:
    pass

model = YOLO(model=model_path_yaml, task='detect')

def train(model, b=yaml_data['batch']):
    return model.train(data=yaml_data['data'], 
                        epochs=200, 
                        imgsz=640, 
                        batch=b,
                        # batch=32,
                        resume=True,
                        name=now,
                        project=out_dirs_path,
                        close_mosaic=30,
                        deterministic=True,
                        cos_lr=True)
import torch

try:    
    result = train(model)
except:
    torch.cuda.empty_cache()
    dir_list = sorted(os.listdir(out_dirs_path), reverse=True)
    for i in dir_list:
        path = os.path.join(out_dirs_path, i, 'weights')
        x = os.listdir(path)
        if x != []:
            model_path_yaml = os.path.join(path, 'last.pt')
            break
    model = YOLO(model=model_path_yaml, task='detect')
    result = train(model, yaml_data['batch']/2)

# results = model.val(data='/home/mamingrui/sod/ultralytics-main/ultralytics/cfg/datasets/VisDrone.yaml', epochs=200, imgsz=640, batch=32)
# result = model.predict(source="/home/mamingrui/sod/visdrone/images/val", device="1", save=True)
with open(out_dirs_path+'/best.txt', 'w') as f:
    f.write(str(result.mean_results()[0])+'\n')
    f.write(str(result.mean_results()[1])+'\n')
    f.write(str(result.mean_results()[2])+'\n')
    f.write(str(result.mean_results()[3]))
import torch
x = torch.rand(1,3,640,640)
xx = model.model(x)
for i in range(3):
    print(xx[i].size())