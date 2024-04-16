# 에셋 드가면 데이터셋 ex안에 있는 이미지와 레이블이 있는데 레이블에는 쓸데없는 어노테이션이 너무많음 이미지 또한 마찬가지
#그래서 지금 존재하는 이미지 말고 다른 어노테이션과 딕셔너리 정보를 다 없애줘
import glob
import json
import copy

def find_filename(dataset_dir):
    a = glob.glob(dataset_dir + "**")
    new_file_list = []
    for i in a:
        file_name = i.split(dataset_dir)
        new_file_list.append(file_name[1])
    return new_file_list
                
def read_json(label_path):
    with open(label_path, "r") as f:
        coco = json.load(f)
    raw_coco = copy.deepcopy(coco)

    return raw_coco

def save_json(coco, coco_dst):
    with open(coco_dst, "w") as outfile:
        json.dump(coco, outfile, indent = 4)

def validate_coco(raw_coco, filenames):
    valid_imgid = []
    new_images = []
    new_annotations = []
    for img in raw_coco["images"]:
        if img["file_name"] in filenames:
            new_images.append(img)
            valid_imgid.append(img["id"])
            
    for ann in raw_coco["annotations"]:
        if ann["image_id"] in valid_imgid:
            new_annotations.append(ann)
    
    del raw_coco["images"], raw_coco["annotations"]
    
    raw_coco["images"] = new_images
    raw_coco["annotations"] = new_annotations
    
    return raw_coco

if __name__ == "__main__":
    dataset_dir = "/home/happyzion/yomce/object-detection-to-classification/assets/datasets_example/images/"
    label_path = "/home/happyzion/yomce/object-detection-to-classification/assets/datasets_example/coco.json"
    coco_dst = "/home/happyzion/yomce/object-detection-to-classification/assets/datasets_example/new_coco.json"

    filenames = find_filename(dataset_dir)
    raw_coco = read_json(label_path)
    new_coco = validate_coco(raw_coco,filenames)
    save_json(new_coco, coco_dst)