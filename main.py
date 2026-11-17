from od2cls.img_process.bbox2img import Bbox2img

new_b2i = Bbox2img(
    coco_file = "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/YonseiPersonDetDataset_v0.1.0/exports/COCO/instances_train.json",
    image_dir = "/home/gpuadmin/repo/KISA/kisa-od-model-trainer/datasets/YonseiPersonDetDataset_v0.1.0/exports/COCO/images",
    imgz = (224, 224)
    )
new_b2i.crop(margin = 0)
# new_b2i.letterbox(aspect_ratio = True)
new_b2i.save_image(folder_per_class = True)