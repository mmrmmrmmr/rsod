from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from show_yolo import draw, shutil
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import datetime
import yaml

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

name = 'cor_yolov8s_baba_c2f_my_c2fchoose_behind_3_256256256'

model_path_yaml = '/home/mamingrui/sod/ultralytics-main/my_model/'+name+'.yaml'
out_dirs_path = '/home/mamingrui/sod/ultralytics-main/out_dirs/'+name

with open(model_path_yaml, 'r', encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)


dir_list = sorted(os.listdir(out_dirs_path), reverse=True)

try:
    for i in dir_list:
        if '.' in i:
             continue
        path = os.path.join(out_dirs_path, i, 'weights')
        x = os.listdir(path)
        if x != []:
            model_path_yaml = os.path.join(path, 'best.pt')
            last_model_path = os.path.join(path, 'last.pt')
            d = os.path.join(out_dirs_path, i)
            break
except:
     print("no found model")


model = YOLO(model=model_path_yaml, task='detect')
# _, ckpt = attempt_load_one_weight(last_model_path)
# print(ckpt['epoch'])

# img_path = '/home/mamingrui/sod/ultralytics-main/visheat'
# img_path = '/home/mamingrui/sod/UAV/test'
img_path = '/home/mamingrui/sod/cors/images/val2017'
# img_path = '/home/mamingrui/sod/VHR/images'

txt_path = '/home/mamingrui/sod/ultralytics-main/output/'+name+'/labels'

path = '/home/mamingrui/sod/ultralytics-main/output/'+name
try:
    shutil.rmtree(path)
except:
    pass
results = model.predict(source=img_path,
                    imgsz=640,
                    name=name,
                    project='output',
                    save_txt=True,
                    exist_ok=True,
                    # save=True,
                    conf=0.40
                    )     
draw(img_path, txt_path)
