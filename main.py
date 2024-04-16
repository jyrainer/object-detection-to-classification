from od2cls.img_process.bbox2img import Bbox2img

new_b2i = Bbox2img(
    coco_file = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\new_coco.json",
    image_dir = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\images"
    )
new_b2i.crop()
print(new_b2i)