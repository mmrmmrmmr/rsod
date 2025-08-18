from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from show_yolo import draw, shutil
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import datetime
import yaml

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

name = 'fai_yolov8s_add128_cs_mid2'

model_path_yaml = '/home/mamingrui/sod/ultralytics-main/my_model/'+name+'.yaml'
out_dirs_path = '/home/mamingrui/sod/ultralytics-main/out_dirs/'+name

with open(model_path_yaml, 'r', encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)


dir_list = sorted(os.listdir(out_dirs_path), reverse=True)

try:
    for i in dir_list:
        path = os.path.join(out_dirs_path, i, 'weights')
        x = os.listdir(path)
        if x != []:
            model_path_yaml = os.path.join(path, 'best.pt')
            last_model_path = os.path.join(path, 'last.pt')
            d = os.path.join(out_dirs_path, i)
            print(d)
            break
except:
     print("no found model")


model = YOLO(model=model_path_yaml, task='detect')
# _, ckpt = attempt_load_one_weight(last_model_path)
# print(ckpt['epoch'])

# img_path = '/home/mamingrui/sod/ultralytics-main/visheat'
img_path = '/home/mamingrui/sod/VHR/images'

txt_path = '/home/mamingrui/sod/yolov5-master/output/'+name+'/labels'

path = '/home/mamingrui/sod/yolov5-master/output/'+name
try:
    shutil.rmtree(path)
except:
    pass

results = model.predict(source='/home/mamingrui/sod/mar20/val.txt',
                    imgsz=640,
                    name=name,
                    project='output',
                    save_txt=True,
                    exist_ok=True,
                    # save=True,
                    conf=0.30
                    )
                    
draw(img_path, txt_path)
