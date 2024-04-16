from od2cls.img_process.bbox2img import Bbox2img

def test_crop_and_letterbox():
    test_instance = Bbox2img(
        coco_file = "assets/datasets_example/new_coco.json",
        image_dir = "assets/datasets_example/images",
        imgz = (224,224)
    )
    assert len(test_instance.coco["images"]) == 40, len(test_instance.coco["categories"] == 8)
    assert test_instance.data == []
    
    test_instance.crop()
    assert test_instance.data != []
    assert len(test_instance.coco["annotations"]) == len(test_instance.data)
    
    test_instance.letterbox()
    for data in test_instance.data:
        assert test_instance.imgz[0] == data["image_data"].shape[1]
        assert test_instance.imgz[1] == data["image_data"].shape[0]