import os

import cv2

from od2cls.coco.validate import read_json, save_json

class Bbox2img:
    def __init__(
        self,
        coco_file,
        image_dir,
        imgz = (224, 224),
        output_dir = "./objects",
        data = []
    ):
        self.coco_file = coco_file
        self.image_dir = image_dir
        self.imgz = imgz
        self.output_dir = output_dir
        self.coco = read_json(self.coco_file)
        self.data = data

    def crop(
        self,
        margin = 0
        ):
        for anns in self.coco["annotations"] :
            image_id = anns["image_id"]                 # 이미지 고유 아이디
            x,y,width,height = map(int,anns["bbox"])    # 박스좌표
            for imgs in self.coco["images"] :
                if image_id == imgs["id"] :
                    raw_file_path = os.path.join(self.image_dir, imgs["file_name"])
                    image = cv2.imread(raw_file_path)
                    
                    if image is not None :
                        new_ymin = max(y - margin, 0)
                        new_ymax = min(y + height + margin, image.shape[0])
                        new_xmin = max(x - margin, 0)
                        new_xmax = min(x + width + margin, image.shape[1])
                        cropped_image = image[new_ymin:new_ymax, new_xmin:new_xmax]
                        new_data = {
                            "file_name" : imgs["file_name"],
                            "raw_file_path" : raw_file_path,
                            "image_data" : cropped_image
                        }
                        self.data.append(new_data)                            
                    else :
                        print(f"{raw_file_path}가 존재하지 않음")
        
    def letterbox(
        self,
        aspect_ratio = True,
    ):
        if aspect_ratio == False:
            for i, data in enumerate(self.data):
                new_arr = cv2.resize(data["image_data"], (self.imgz[0], self.imgz[1]), interpolation=cv2.INTER_LINEAR)
                self.data[i]["image_data"] = new_arr
        elif aspect_ratio == True:
            for i, data in enumerate(self.data):
                if data["image_data"].shape[1] >= data["image_data"].shape[0]: # width 가 더 큰경우 
                    ratio = self.imgz[0] / data["image_data"].shape[1]
                elif data["image_data"].shape[1] < data["image_data"].shape[0]:
                    ratio = self.imgz[1] / data["image_data"].shape[0]
                dw, dh = (int(data["image_data"].shape[1] * ratio), int(data["image_data"].shape[0] * ratio ))
                new_arr = cv2.resize(data["image_data"], (dw, dh), cv2.INTER_LINEAR)
                
                dw = self.imgz[0] - dw
                dh = self.imgz[1] - dh
                
                dw /= 2
                dh /= 2
                
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                
                new_arr = cv2.copyMakeBorder(new_arr, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
                self.data[i]["image_data"] = new_arr
    
    def save_image(
        self
    ):
        os.makedirs(self.output_dir, exist_ok= True)
        for data in self.data:
            result_path = os.path.join(self.output_dir, data["file_name"])
            cv2.imwrite(result_path, data["image_data"])