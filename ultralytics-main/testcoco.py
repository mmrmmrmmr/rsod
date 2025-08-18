from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json, os

# 加载coco_instances_results.json文件中的预测结果
def cocoac(ins_path, annFile=None, name=None):
    root = "." # 这里修改为你得到的val文件路径
    # ins_path = "/home/mamingrui/sod/ultralytics-main/out_dirs/yolov8m/20240831_132347/test/20240901_184210/predictions.json"
    ins_path_S = ins_path.split("/")
    print(ins_path)
    with open(ins_path, 'r') as f:
        results = json.load(f)

    # 加载COCO数据集的注释文件
    if annFile == None:
        annFile = '/home/mamingrui/sod/visdrone/VisDrone2019-DET-val/VisDrone2019-DET_val_coco_2cut.json' # 这里加载你的验证集的json数据集
    cocoGt = COCO(annFile)

    # 加载预测结果到COCO格式中
    cocoDt = cocoGt.loadRes(results)

    # 初始化COCO评估器
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
    with open(name, 'a') as f:
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

    # print("AP、MR、DR和AR指标已保存到{}/COCO_AP_MR_DR_AR.txt文件中。".format(root))
