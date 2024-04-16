import os
import json
import copy

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
        self.data = []
        
    def crop(
        self
        ):
        
        coco_json = read_json(self.coco_file)

        for anns in coco_json["annotations"] :
            image_id = anns["image_id"]                 # 이미지 고유 아이디
            x,y,width,height = map(int,anns["bbox"])    # 박스좌표
            for imgs in coco_json["images"] :
                if image_id == imgs["id"] :
                    raw_file_path = os.path.join(self.image_dir, imgs["file_name"])
                    image = cv2.imread(raw_file_path)
                    
                    if image is not None :
                        cropped_image = image[y:y+height, x:x+width]
                        new_data = {
                            "file_name" : imgs["file_name"],
                            "raw_file_path" : raw_file_path,
                            "image_data" : cropped_image
                        }
                        self.data.append(new_data)
                        #cv2.imwrite(output_path + 'nzia_' + ".jpg",cropped_image)
                    else :
                        print(f"{raw_file_path}가 존재하지 않음")
                    
        
        
if __name__ == "__main__" :
    new_b2i = Bbox2img(
        coco_file = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\new_coco.json",
        image_dir = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\images"
        )
    