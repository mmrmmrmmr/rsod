import os
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
 
FONT_SIZE = 13*2
# IMAGE_FONT = ImageFont.truetype(u"simhei.ttf", FONT_SIZE)
COLOR_LIST = ["red", "green", "blue", "cyan", "yellow", "purple",
              "deeppink", "ghostwhite", "darkcyan", "olive",
              "orange", "orangered", "darkgreen"]
 
 
class ShowResult(object):
    def __init__(self, args):
        self.args = args
        self.json_file_path = self.args.json_file_path
        self.img_path_root = self.args.img_path_root
        self.show_img_path_root = self.args.show_img_path_root
        self._category_id_convert_to_name()
        self._data_preprocess()
 
    def _category_id_convert_to_name(self):
        # 构建一个字典: {"category_id": category_name,...}
        orig_json_file = open(self.json_file_path, encoding='utf-8')
        self.orig_json = json.load(orig_json_file)
        self.category_id_name_dict = {}
        orig_category_id_name_dict = self.orig_json["categories"]
        for idx, category_id_name_dict in enumerate(orig_category_id_name_dict):
            self.category_id_name_dict[category_id_name_dict["id"]] = category_id_name_dict["name"]
 
    def _data_preprocess(self):
        # 构建一个字典: {"img_id": [{bbox_attr_dict}, ...], ...}
        result_json_file = open(self.json_file_path, encoding='utf-8')
        self.resut_json = json.load(result_json_file)
        self.resut_json = self.resut_json["annotations"]
        self.img_id_bboxes_attr_dict = {}
        for idx, result_ann in enumerate(self.resut_json):
            result_attr_dict = {"bbox": result_ann["bbox"],
                                #"score": result_ann["score"],
                                "category_id": result_ann["category_id"]}
            if result_ann["image_id"] not in self.img_id_bboxes_attr_dict.keys():
                self.img_id_bboxes_attr_dict[result_ann["image_id"]] = []
                self.img_id_bboxes_attr_dict[result_ann["image_id"]].append(result_attr_dict)
            else:
                self.img_id_bboxes_attr_dict[result_ann["image_id"]].append(result_attr_dict)
 
    def mainprocess(self):
        # 对 self.img_id_bboxes_attr_dict进行循环操作:
        # 对每一张图片
        for idx, (img_id, attr_dict_list) in enumerate(self.img_id_bboxes_attr_dict.items()):
            img_path = os.path.join(self.img_path_root, img_id + ".png")
            print("当前正在处理第-{0}-张图片, 总共需要处理-{1}-张, 完成百分比:{2:.2%}".format(idx+1,
                                                                        len(self.img_id_bboxes_attr_dict.keys()),
                                                                        (idx+1) / len(self.img_id_bboxes_attr_dict.keys())))
            # 对每一个bbox标注
            img = Image.open(img_path, "r")  # img1.size返回的宽度和高度(像素表示)
            draw = ImageDraw.Draw(img)
            # 提取所有bboxes信息
            for jdx, attr_dict in enumerate(attr_dict_list):
                COLOR = COLOR_LIST[jdx % len(COLOR_LIST)]
                # 对每一个bboxes标注信息
                bbox = attr_dict["bbox"]
                #score = attr_dict["score"]
                category_name = self.category_id_name_dict[attr_dict["category_id"]]
                x1, y1 = bbox[0], bbox[1]
 
                top_left = (int(bbox[0]), int(bbox[1]))  # x1,y1
                top_right = (int(bbox[0])+int(bbox[2]), int(bbox[1]))  # x2,y1
                down_right = (int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3]))  # x2,y2
                down_left = (int(bbox[0]), int(bbox[1])+int(bbox[3]))  # x1,y2
 
                draw.line([top_left, top_right, down_right, down_left, top_left], width=5, fill=COLOR)
 
                # 将类别和分数写在左上角
                #new_score = str(score).split('.')[0] + '.' + str(score).split('.')[1][:2]
                #draw.text((x1, y1 - FONT_SIZE), new_score, font=IMAGE_FONT, fill=COLOR)
                #draw.text((x1 + 25, y1 - FONT_SIZE), "|", font=IMAGE_FONT, fill=COLOR)font=IMAGE_FONT, 
                draw.text((x1 + 30, y1-FONT_SIZE), str(category_name), fill=COLOR)
 
            # 存储图片
            save_path = os.path.join(self.show_img_path_root, img_id + ".png")
            img.save(save_path, 'png')
 
 
def main():
    parser = argparse.ArgumentParser()
    # json文件路径
    parser.add_argument('-json_file_path', default="/home/mamingrui/sod/ultralytics-main/output/yolov8m/predictions.json",
                        help='the single img json file path')
    # 图片文件夹root
    parser.add_argument('-img_path_root', default="/home/mamingrui/sod/ultralytics-main/visheat",
                        help='the val img path root')
 
    # 显示图片存储文件路径
    parser.add_argument('-show_img_path_root', default="t",
                        help='the show img path root')
 
    args = parser.parse_args()
    showresult = ShowResult(args)
 
    showresult.mainprocess()
 
 
if __name__ == '__main__':
    main()