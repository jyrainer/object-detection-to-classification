
import json
import random
import os
import cv2
import yaml
import tqdm
import shutil

# 1. 기존 데이터셋 3개에 대해, split 키를 부여하여 train/val/test 중 하나로 세팅 (0.8, 0.1, 0.1)

def split_datasets(coco, output_json_path, split=(0.8, 0.1, 0.1), method="random"):
    # 예외 처리
    if sum(split) != 1.0:
        raise ValueError("Error: split - 입력값의 합이 1이어야 합니다")
    
    if method != "random":
        raise ValueError("Error: method - 예측되지 않은 입력값입니다. (가능한 method: ['random'])")

    images = coco.get("images", [])

    num_images = len(images)
    num_train = int(num_images * split[0])
    num_val = int(num_images * split[1])
    num_test = num_images - (num_train + num_val)

    indices = list(range(num_images))
    random.shuffle(indices)

    train_indices = set(indices[:num_train])
    val_indices = set(indices[num_train:num_train + num_val])
    test_indices = set(indices[num_train + num_val:])

    for i, image in enumerate(images):
        if i in train_indices:
            image["split"] = "train"
        elif i in val_indices:
            image["split"] = "val"
        else:
            image["split"] = "test"

    # json 파일 저장
    output_json_path = os.path.join(output_json_path, "instance_Train_add_split.json")

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=4, ensure_ascii=False)

    return coco

# 2. extract하여 cls 학습 가능하도록 모두 저장 (PersonCOCODetDataset_v0.2.0.zip는 크롭할 필요 없음!)

def export_classify_datasets(coco, save_path, image_dir):
    # 기존 폴더가 있을 경우 삭제 후 새로 실행하도록 에러 띄우기
    if os.path.exists(save_path):
        raise Exception(f"Error: '{save_path}' 경로에 이미 데이터가 존재합니다. 다른 폴더를 지정하거나 기존 폴더를 삭제 후, 다시 시도해 주세요.")
    
    os.makedirs(save_path, exist_ok=True)

    # train, val, test 이름 지정 및 class categories 정보 불러와서 저장
    splits = ["train", "val", "test"]
    class_names = {}
    categories = coco.get("categories", [])
    for category in categories:
        class_names[category["id"]] = category["name"]

    # 모든 split(train,val,test)에 대해 각 class categories 폴더 생성
    for split in splits:
        split_path = os.path.join(save_path, split)
        os.makedirs(split_path, exist_ok=True)

        for class_name in class_names.values():
            os.makedirs(os.path.join(split_path, class_name), exist_ok=True)

    # image_id: (파일명, split 값) 딕셔너리 생성
    image_id_to_info = {
        img["id"]: (img["file_name"], img.get("split", "train"))
        for img in coco.get("images", [])
    }

    # Annotation 정보를 활용해 이미지 크롭 및 저장
    annotations = coco.get("annotations", [])
    
    for ann in tqdm.tqdm(annotations):
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        bbox = ann["bbox"]  # [x, y, width, height]
        
        # 이미지 파일명 & 이미지 split 값 가져오기
        image_filename = image_id_to_info[image_id][0]
        split = image_id_to_info[image_id][1]
        
        # 카테고리 이름 가져오기
        class_name = class_names.get(category_id, "unknown")

        # 저장할 경로 설정
        class_folder = os.path.join(save_path, split, class_name)

        # 원본 이미지 경로 설정
        image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue  # 원본 이미지 없으면 스킵

        img = cv2.imread(image_path)

        if img is None:
            print(f"Warning: Cannot read image {image_path}") # 이미지가 안 읽히면 스킵
        else:
            x, y, w, h = map(int, bbox)
            cropped_img = img[y:y+h, x:x+w]
            cropped_filename = f"{image_filename.split('.')[0]}_{image_id}_{ann['id']}_{class_name}.jpg"
            cv2.imwrite(os.path.join(class_folder, cropped_filename), cropped_img, [cv2.IMWRITE_JPEG_QUALITY, 95]) # 품질(95)은 클수록 좋음(0~100)

    # data.yaml 생성
    data_yaml = {
        "path": save_path.replace("\\", "/"),
        "train": "train",
        "val": "val",
        "test": "test",
        "names": list(class_names.values())
    }

    with open(os.path.join(save_path, "data.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Dataset successfully exported to {save_path}")

# 3. 기존에 있던 person_helmet_dataset으로 파일 모두 복사하여 옮기기


if __name__ == "__main__":

    ### coco dataset으로부터 Cls 디렉토리 형식으로 변환 #####
    root_path_list = [
        "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/YonseiPersonFalseDetDataset_v0.1.0/exports/COCO"
    ]
        
    for root_path in root_path_list: 
        input_json_path = f"{root_path}/instances_train.json"  # CVAT에서 받은 원본 JSON 경로
        save_path = "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/YonseiPersonFalseClsDataset_v0.1.0" # root 파일명 + "_Cls"
        image_dir= f"{root_path}/images" # CVAT에서 받은 원본 이미지 경로로

        with open(input_json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        splited_coco = split_datasets(coco, output_json_path=root_path, split=(0.8, 0.1, 0.1), method="random")

        export_classify_datasets(splited_coco, save_path, image_dir)