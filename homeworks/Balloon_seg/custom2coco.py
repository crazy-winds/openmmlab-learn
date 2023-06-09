import os
import mmcv
from itertools import chain


def cutom2coco(json_path, img_prefix_path, target_path="test.json"):
    """ 自定义数据集转COCO数据集格式

    Args:
        json_path (str): 自定义数据ann_file路径
        img_prefix_path (str): 图片路径的前缀
        target_path (str): 生成COCO格式数据集保存

    COCO = {
        "images": [
            {
                "id": 0,
                "file_name": "1.jpg",
                "height": 123,
                "width": 123
            },
            ...
        ],
        "annotations": [
            {
                "image_id": 0,
                "id": 0,
                "category_id": 1,
                "bbox": [0, 0, 4, 4],
                "area": 16,
                "segmentation": [0, 0, 1, 1, 2, 2, 3, 3, ...],
                "iscrowd": 0 if "polygon" else 1
            },
            ...
        ],
        "categories": [
            {
                "id": 1,
                "name": "lemon",
            },
            ...
        ]
    }
    """
    cutom_json = mmcv.load(json_path)
    coco = {"categories": [{"id": 0, "name": "balloon"}]}
    annotations = []
    images = []
    ann_idx = 0

    for item in cutom_json.values():
        img_path = item["filename"]
        img_id = img_path[:-4]
        h, w, c = mmcv.imread(os.path.join(img_prefix_path, img_path)).shape
        
        regions = item["regions"]
        for key, attr in regions.items():
            x_min, x_max, y_min, y_max = (
                min(attr["shape_attributes"]["all_points_x"]),
                max(attr["shape_attributes"]["all_points_x"]),
                min(attr["shape_attributes"]["all_points_y"]),
                max(attr["shape_attributes"]["all_points_y"]),
            )

            annotations.append({
                "image_id": img_id,
                "id": ann_idx,
                "category_id": 0,
                "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                "area": (x_max - x_min) * (y_max - y_min),
                "segmentation": [list(chain.from_iterable(zip(attr["shape_attributes"]["all_points_x"], attr["shape_attributes"]["all_points_y"])))],
                "iscrowd": int(attr["shape_attributes"]["name"] == "rel")
            })
            ann_idx += 1

        images.append({
            "id": img_id,
            "file_name": img_path,
            "height": h,
            "width": w
        })

    coco["annotations"] = annotations
    coco["images"] = images

    mmcv.dump(coco, target_path, indent=4)

# train
cutom2coco(
    json_path="./data/balloon/train/via_region_data.json",
    img_prefix_path="./data/balloon/train",
    target_path="./data/balloon/train/coco_train.json"
)

# val
cutom2coco(
    json_path="./data/balloon/val/via_region_data.json",
    img_prefix_path="./data/balloon/val",
    target_path="./data/balloon/val/coco_val.json"
)