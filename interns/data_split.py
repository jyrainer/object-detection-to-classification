import json
import random
from collections import defaultdict

def split_datasets(COCO, split=(0.8,0.1,0.1), method='random'):

    images = COCO['images']
    num_images = len(COCO['images'])
    print(num_images)
    
    
    if sum(split) != 1.0:
        raise ValueError('합이 반드시 1이어야 함')
    
    if method != 'random':
        raise ValueError('random으로 입력해주세요')

    train_size = int(num_images * split[0])
    val_size = int(num_images * split[1])
    test_size = int(num_images * split[2])

    print(train_size, val_size, test_size)

    if method=='random':
        random.shuffle(images)

    train_set = images[:train_size]
    val_set = images[train_size:train_size + val_size]
    test_set = images[train_size + val_size:]

    for image in train_set:
        image['split'] = 'train'
    for image in val_set:
        image['split'] = 'val'
    for image in test_set:
        image['split'] = 'test'

    #파일 새로 저장
    file_save = '/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/YonseiPersonDetDataset_v0.1.0/exports/COCO/instances_train_new.json'
    with open(file_save, "w", encoding="utf-8") as f:
        json.dump(COCO, f, ensure_ascii=False, indent=4)

    print("저장 완료")

    print(f"image num : {len(COCO['images'])}")
    print(f"annotations num : {len(COCO['annotations'])}")

    # 카테고리별 Annotation 개수 카운트
    category_count = defaultdict(int)
    for ann in COCO['annotations']:
        category_count[ann["category_id"]] += 1

    print("\n📌 카테고리별 Annotation 개수:")
    for cat in COCO['categories']:
        cat_id = cat["id"]
        cat_name = cat["name"]
        print(f"  - {cat_name} (ID: {cat_id}): {category_count[cat_id]} 개")

    
    return COCO


file_path = '/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/YonseiPersonDetDataset_v0.1.0/exports/COCO/instances_train.json'

with open(file_path, 'r', encoding='utf-8') as f:
    COCO = json.load(f)
    print(type(COCO))
    print(COCO.keys())
    print('images' in COCO)
    
split_datasets(COCO, split=(0.8,0.1,0.1),method='random')


#딕셔너리같은 건 예민해서, 원본 파일을 import copy해서
#dpcopy해서 하는게 안전. 원본 coco에 접근하는 것보단,,