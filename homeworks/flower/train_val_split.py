import os
import shutil
import random


def data_split(root_path, train_rate=.8, seed=3407, target_path="./dataset/flower_dataset"):
    # 获取所有类别
    classes = os.listdir(root_path)
    with open(os.path.join(target_path, "classes.txt"), "w") as f:
        f.write("\n".join(classes))

    # 创建训练验证文件夹
    for name in ("train", "val"):
        if not os.path.exists(os.path.join(target_path, name)):
            os.mkdir(os.path.join(target_path, name))

    # 划分数据集
    train_list = []
    valid_list = []
    for cls in classes:
        cur_cls = os.listdir(os.path.join(root_path, cls))
        cur_cls_length = len(cur_cls)
        random.seed(seed)
        random.shuffle(cur_cls)
        spt = int(cur_cls_length * train_rate)

        # 训练
        src_root_path = os.path.join(root_path, cls)
        if not os.path.exists(os.path.join(target_path, "train", cls)):
            os.mkdir(os.path.join(target_path, "train", cls))
        for i in range(spt):
            train_list.append(f"train/{cls}/{cur_cls[i]}")
            # 创建软连接
            shutil.copyfile(
                os.path.join(src_root_path, cur_cls[i]),
                os.path.join(target_path, "train", cls, cur_cls[i])
            )

        # 验证
        if not os.path.exists(os.path.join(target_path, "val", cls)):
            os.mkdir(os.path.join(target_path, "val", cls))
        for i in range(spt, cur_cls_length):
            train_list.append(f"val/{cls}/{cur_cls[i]}")
            # 创建软连接
            shutil.copyfile(
                os.path.join(src_root_path, cur_cls[i]),
                os.path.join(target_path, "val", cls, cur_cls[i])
            )

        print(
            f"{cls} have {cur_cls_length},"
            f"Train data have {spt}, "
            f"Valid data have {cur_cls_length - spt}"
        )


    with open(os.path.join(target_path, "train.txt"), "w") as f:
        f.write("\n".join(train_list))

    with open(os.path.join(target_path, "val.txt"), "w") as f:
        f.write("\n".join(valid_list))

data_split("data/flower_dataset")