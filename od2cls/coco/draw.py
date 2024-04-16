from pycocotools.coco import COCO
import os
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
 

def draw_annotations(coco_file, image_dir, dst_dir):
    annotation_file = os.path.join(coco_file)
    coco = COCO(annotation_file)
    os.makedirs(dst_dir, exist_ok=True)
    
    for img_id in tqdm(coco.getImgIds()):
        try:
            cond = 0
            # 이미지 정보 가져오기
            image_info = coco.loadImgs(img_id)[0]
            image_path = os.path.join(image_dir, image_info['file_name'])
            image = Image.open(image_path)
            
            # 이미지에 대한 주석 정보 가져오기
            annotation_ids = coco.getAnnIds(imgIds=img_id)
            annotations = coco.loadAnns(annotation_ids)
    
            cond2 = (image_info['height'] != image.size[1]) or (image_info['width'] != image.size[0])
            if cond2:
                cond = 1

            draw = ImageDraw.Draw(image)
            for annotation in annotations:
                bbox = annotation['bbox']
    
                cond1 = (bbox[0] > image.size[0]) or (bbox[1] > image.size[1])
                cond3 = (bbox[0]+bbox[2]) > image.size[0] or (bbox[1]+bbox[3])> image.size[1]
                if cond1:
                    cond = 1
                category = coco.loadCats(annotation['category_id'])[0]['name']
                outline_color = 'red'
                line_width = 2
                x0, y0, width, height = bbox
                x1, y1 = x0 + width, y0 + height
                draw.line([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], fill=outline_color, width=line_width)
                draw.text((x0, y0 - 20), category, fill=outline_color)
            
            # 이미지 저장
            if cond:
                pass
            else:
                output_path = os.path.join(dst_dir, image_info['file_name'])
                path = Path(output_path).parent
                path.mkdir(parents=True, exist_ok=True)
                image.save(output_path)
        except:
            pass
        
    print("Annotations for all images have been saved.")
        
if __name__ == "__main__":
    coco_file = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\new_coco.json"
    image_dir = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\images"
    dst_dir = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\draw_images"
    draw_annotations(coco_file, image_dir, dst_dir)