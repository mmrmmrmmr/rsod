from ultralytics import YOLO
from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import datetime
import yaml

import sys
import testcoco
now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

name_list = [
    #  'yolov8s_scam_my3bone2l_3bicross4neck256256256_4fem_my3head',
    #  'yolov8s_baba_c2f256256256',
    #  'yolov8s_baba_c3256256256',
    #  'yolov8s_baba_bif4neck256256256',
        # 'yolov8s_baba_c2fchoose_behind256256256'
    #  'yolov8s_scam_my3bone2l_3bicross4neck256256256',
    #  'yolov8s_scam_my3bone2l',
    #  'vhr_yolov8s_baba_c2fchoose_behind_3_256256256',
    #  'vhr_yolov8s_baba_c2f_my_c2fchoose_behind_3_256256256',
    #  'vhr_yolov8s_baba_c2f_my_256256256',
    #  'uav_yolov8s_baba_c2fchoose_behind_3_256256256',
    #  'vis_yolov8s_ffmmyhead',
    #  'uav_yolov8s_baba_c2f_my_c2fchoose_behind_3_256256256',
    #  'vis_yolov8s_baba_mychoose_behind_2_256256256',
    #  'vis_yolov8s_baba_mychoose_behind_2_256256256_lossabc1',
    #  'vis_yolov8s_baba_mychoose_behind_2_256256256_ffmmyhead',
    #  'uav_yolov8s_baba_mychoose_behind_2_256256256_ffmmyhead_lossabc1',
    #  'vis_yolov8s_ffmmyhead_lossabc1',
    #  'vis_yolov8s_lossabc1',
    #  'vis_yolov8s_fem3_front_lossabc',
    #  'vis_yolov8s_fem3_front',
    #  'vis_yolov8s_baba_4head_mid4',
     'fai_yolov8s_add128_cs_mid2',
    #  'fai_yolov8m',
    #  'vis_yolov8s_baba_fem3_front_4head_mid4_lossgiou',
    #  'vis_yolov8s_baba_fem3_front_4head_mid4_lossdiou',
    #  'vis_yolov8s_baba_fem3_front_4head_mid4',
    #  'uav_yolov8s_baba_fem3_front_4head_mid4_lossabc',
    #  'vhr_yolov8s_scam_my3bone2l_3bicross4neck256256256_4fem_my3head',
    #  'uav_yolov8s_scam_my3bone2l_3bicross4neck256256256_4fem_my3head',
    #  'yolov8s_fem_my3head',
    #  'yolov8s_baba_3bicross4neck256256256_4fem_my3head',
    #  'mar_yolov8s',
    #  'yolov8s5',
    #  'vhr_yolov6m',
    #  'yolov8s6b',
]

def m(name=None):
    
    model_path_yaml = '/home/mamingrui/sod/ultralytics-main/my_model/'+name+'.yaml'
    out_dirs_path = '/home/mamingrui/sod/ultralytics-main/out_dirs/'+name

    with open(model_path_yaml, 'r', encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)


    dir_list = sorted(os.listdir(out_dirs_path), reverse=True)

    try:
        for i in dir_list:
            print(i)
            if '.' in i:
                 continue
            path = os.path.join(out_dirs_path, i, 'weights')
            print(path)
            x = os.listdir(path)
            if x != []:
                model_path_yaml = os.path.join(path, 'best.pt')
                # model_path_yaml = os.path.join(path, 'last.pt')
                d = os.path.join(out_dirs_path, i)
                break
    except:
        print("no found model")
        return
    # str(now)+
    txt = "/home/mamingrui/sod/ultralytics-main"+'/out_test/'+name+'.txt'
    sys.stdout = open(txt, 'w')

    model = YOLO(model=model_path_yaml, task='detect')

    print('                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)')
    results = model.val(data=yaml_data['data'],
                        imgsz=640,
                        name=now,
                        project=d+'/test/',
                        save_json=True
                        )
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    testcoco.cocoac(d+'/test/'+str(now)+'/predictions.json', name=txt, annFile='/home/mamingrui/sod/visdrone/VisDrone2019-DET-val/VisDrone2019-DET_val_coco_2cut.json')
# 
err = []

for n in name_list:
    m(n)
    # try:
    #     m(n)
    # except:
    #     err.append(n)
print(err)

