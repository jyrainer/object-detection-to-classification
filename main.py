from od2cls.img_process.bbox2img import Bbox2img

new_b2i = Bbox2img(
    coco_file = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\new_coco.json",
    image_dir = "C:\\Users\\admin\\Desktop\\Github\\object-detection-to-classification\\assets\\datasets_example\\images",
    imgz = (224, 224)
    )
new_b2i.crop(margin = 100)
new_b2i.letterbox(aspect_ratio = True)
new_b2i.save_image(folder_per_class = True)