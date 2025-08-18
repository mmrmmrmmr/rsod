import json
import os
from collections import OrderedDict
from testcoco import *

# 获取标注文件图像id与图像名字的字典
def get_name2id_map(image_dict):
    name2id_dict = OrderedDict()
    for image in image_dict:
        file_name = image['file_name'].split('.')[0]    # maksssksksss98.png -> maksssksksss98
        id = image['id']
        name2id_dict[file_name] = id
    return name2id_dict

if __name__ == '__main__':
    anno_json = '/home/mamingrui/sod/UAV/test.json'
    pred_json = '/home/mamingrui/sod/ultralytics-main/out_dirs/uav_yolov8s_scam_my3bone2l_3bicross4neck256256256_4fem_my3head/20240919_141636/test/20240919_172812/predictions.json'
    with open(pred_json, 'r') as fr:
        pred_dict = json.load(fr)
    with open(anno_json, 'r') as fr:
        anno_dict = json.load(fr)
    name2id_dict = get_name2id_map(anno_dict['images'])
    # 对标注文件annotations的image_id进行更改
    # for annotations in anno_dict['annotations']:
    #     image_id = annotations['image_id']
    #     annotations['image_id'] = int(name2id_dict[image_id])
    # 对预测文件的image_id同样进行更改
    for predictions in pred_dict:
        image_id = predictions['image_id']
        predictions['image_id'] = int(name2id_dict[image_id])
    # 分别保存更改后的标注文件和预测文件
    with open('anno_json.json', 'w') as fw:
        json.dump(anno_dict, fw, indent=4, ensure_ascii=False)
    with open('pred_json.json', 'w') as fw:
        json.dump(pred_dict, fw, indent=4, ensure_ascii=False)
    cocoac('anno_json.json', 'pred_json.json', 'xxxx')
    cocoGt = COCO('anno_json.json')
    cocoDt = cocoGt.loadRes(pred_dict)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    # 运行评估计算AP
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # 输出平均精度（AP）
    print("Average Precision (AP): {:.2f}".format(cocoEval.stats[0]))
    # 输出漏检率和检测率指标
    print("Miss Rate (MR): {:.2f}".format(1 - cocoEval.stats[8]/cocoEval.stats[2]))
    print("Detection Rate (DR): {:.2f}".format(cocoEval.stats[8]/cocoEval.stats[2]))
    # 将输出结果保存到txt文件中
    with open('xxxx', 'a') as f:
        f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[0]))
        f.write("Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[1]))
        f.write("Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[2]))
        f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[3]))
        f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[4]))
        f.write("Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[5]))
        f.write("\n")
        f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = {:.3f}\n".format(cocoEval.stats[6]))
        f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = {:.3f}\n".format(cocoEval.stats[7]))
        f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[8]))
        f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[9]))
        f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[10]))
        f.write("Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = {:.3f}\n".format(cocoEval.stats[11]))
        f.write("\n")
        f.write("Miss Rate (MR): {:.2f}\n".format(1 - cocoEval.stats[8]/cocoEval.stats[2]))
        f.write("Detection Rate (DR): {:.2f}\n".format(cocoEval.stats[8]/cocoEval.stats[2]))
