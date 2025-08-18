import os
from PIL import Image
from torch.utils.data import Dataset
from data.transforms import LeopartTransforms

class DataSet_iSAID_Pretrain(Dataset):
    def __init__(self, args, path):
        self.transform = LeopartTransforms(size_crops=args['transform']['size_crops'],
                                           nmb_crops=args['transform']['nmb_samples'],
                                           min_scale_crops=args['transform']["min_scale_crops"],
                                           max_scale_crops=args['transform']["max_scale_crops"],
                                           min_intersection=args['transform']["min_intersection_crops"],
                                           jitter_strength=args['transform']["jitter_strength"],
                                           blur_strength=args['transform']["blur_strength"]
                                           )
        # self.data_listn = '/home/mamingrui/sod/VHR/images/' # The list of image paths '/home/mamingrui/sod/visdrone/VisDrone2019-DET-train/images/'
        
        self.data_listn = path
        data_list = os.listdir(self.data_listn)
        self.data_list = []
        import random
        for i in range(len(data_list)):
            # print(data_list[i])
            if data_list[i].endswith(".jpg"):
                self.data_list.append(self.data_listn + data_list[i])
                # print(self.data_list[-1])
        if len(data_list) > 1000:
            self.data_list = random.sample(self.data_list, 1000)
        print('Read ' + str(len(self.data_list)) + ' images')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        img_path = self.data_list[item]
        img = Image.open(img_path)
        multi_crops, gc_bboxes, otc_bboxes, flags = self.transform(img)
        return multi_crops, gc_bboxes, otc_bboxes, flags
