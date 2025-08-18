import cv2
import os
import shutil

COLOR_LIST = [
(220, 20, 60), (119, 11, 32), (100, 250, 0), (0, 0, 230), (106, 0, 228),
(0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
(100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
(165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
(0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
(199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
(209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
(92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
(174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
(255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
(207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
(74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
(0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
(227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
(163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
(183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
(166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
(65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
(196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
(246, 0, 122), (191, 162, 208)    
]

def draw_box_in_single_image(image_path, txt_path, name, label_folder):
    # 读取图像
    image = cv2.imread(image_path)

    # 读取txt文件信息
    def read_list(txt_path):
        pos = []
        try:
            with open(txt_path, 'r') as file_to_read:
                while True:
                    lines = file_to_read.readline()  # 整行读取数据
                    if not lines:
                        break
                    # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
                    p_tmp = [float(i) for i in lines.split(' ')]
                    pos.append(p_tmp)  # 添加新读取的数据
                    # Efield.append(E_tmp)
                    pass
        except:
            pos = [[0,0,0,0,0]]
        return pos


    # txt转换为box
    def convert(size, box):
        xmin = (box[1]-box[3]/2.)*size[1]
        xmax = (box[1]+box[3]/2.)*size[1]
        ymin = (box[2]-box[4]/2.)*size[0]
        ymax = (box[2]+box[4]/2.)*size[0]
        box = (int(xmin), int(ymin), int(xmax), int(ymax))
        return box

    pos = read_list(txt_path)
    # print(pos)
    tl = int((image.shape[0]+image.shape[1])/2)
    lf = max(tl-1,1)
    x = image.copy()
    for i in range(len(pos)):
        label = str(int(pos[i][0]))
        # print(pos[i],image.shape)
        # print('label is '+label)
        box = convert(image.shape, pos[i])
        # print(box)
        image = cv2.rectangle(image,(box[0], box[1]),(box[2],box[3]),COLOR_LIST[int(label)],-1)
        # cv2.putText(image,label,(box[0],box[1]-2), 0, 1, [0,0,255], thickness=2, lineType=cv2.LINE_AA)

    alpha = 0.7
    gamma = 0
    image = cv2.addWeighted(x,alpha,image,1-alpha,gamma)
    for i in range(len(pos)):
        label = str(int(pos[i][0]))
        # print(pos[i],image.shape)
        # print('label is '+label)
        box = convert(image.shape, pos[i])
        # print(box)
        image = cv2.rectangle(image,(box[0], box[1]),(box[2],box[3]),COLOR_LIST[int(label)],1)
    
        # cv2.putText(image,label,(box[0],box[1]-2), 0, 1, [0,0,255], thickness=2, lineType=cv2.LINE_AA)
    print(label_folder+'/image/'+name)
    cv2.imwrite(label_folder+'/image/'+name+'.jpg', image)


def draw(img_folder, label_folder):

    
    # img_folder = r"E:\visdrone\VisDrone2019-DET-train\images"
    img_list = os.listdir(img_folder)
    # img_list.sort()
    path = label_folder+'/image/'
    # try:
    #     shutil.rmtree(path)
    # except:
    #     pass
    if not os.path.exists(path):
        os.makedirs(path)
    # print(label_list)
    # label_list.sort()
    for i in range(len(img_list)):
        image_path = img_folder + "/" + img_list[i]
        txt_path = label_folder + "/" + img_list[i].split('.')[0]+'.txt'
        # txt_path = label_folder + "\\" + label_list[i]
        draw_box_in_single_image(image_path, txt_path, img_list[i].split('.')[0], label_folder)

if __name__ == '__main__':
    label_folder = "/home/mamingrui/sod/ultralytics-main/output/yolov8m/labels"
    img_folder = "/home/mamingrui/sod/ultralytics-main/visheat"
    draw(img_folder, label_folder)
    