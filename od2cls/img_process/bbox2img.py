import os
import cv2

from od2cls.coco.validate import read_json

class Bbox2img:
    """
    coco annotations format 데이터의 bbox 정보를 이용하여
    분류 모델 데이터셋 확보를 위한 클래스이다.
    어노테이션을 crop 후 letterbox를 생성하여 이미지 정보를 획득할 수 있다.
    save_image 메서드를 통해 이미지를 저장할 수 있다.
    """
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
        """
        컨셉에 따라 인스턴스 생성 시 필수적으로 사용해야할 메서드
        crop된 이미지의 array정보를 가진 **data**를 얻을 수 있다.

        Args:
            margin (int, optional): 마진 픽셀 수를 지정할 수 있다. Defaults to 0.
        """
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
                            "category_id" : anns["category_id"],
                            "image_data" : cropped_image
                        }
                        self.data.append(new_data)                            
                    else :
                        print(f"{raw_file_path}가 존재하지 않음")
        
    def letterbox(
        self,
        aspect_ratio = True,
    ):
        """
        레터박스를 만들 수 있는 기본적인 코드 구현

        Args:
            aspect_ratio (bool, optional): 비율 유지 여부. Defaults to True.
        """
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
    
    @staticmethod
    def find_category_name(coco):
        """
        Args:
            coco (dict): coco annotations format

        Returns:
            cate_dict (dict): 카테고리 id와 카테고리 이름 쌍으로 된 딕셔너리
        """
        cate_dict = dict()
        for cate in coco["categories"]:
            cate_dict[cate["id"]] = cate["name"]
        return cate_dict
        
    def save_image(
        self,
        folder_per_class = True
    ):
        """

        Args:
            folder_per_class (bool, optional): 클래스 별 폴더를 생성하여 해당하는 폴더에 이미지를 저장. Defaults to True.
        """
        os.makedirs(self.output_dir, exist_ok= True)
        if folder_per_class == True:
            category_info = Bbox2img.find_category_name(self.coco)
            for _, v in category_info.items():
                cate_dir = os.path.join(self.output_dir, v)
                os.makedirs(cate_dir, exist_ok=True)
            
        for data in self.data:
            if folder_per_class == False:
                result_path = os.path.join(self.output_dir, data["file_name"])
            else:
                result_path = os.path.join(self.output_dir, category_info[data["category_id"]], data["file_name"])
            cv2.imwrite(result_path, data["image_data"])