import mmcv
import cv2 as cv
import numpy as np
from mmdet.apis import inference, inference_detector


video = mmcv.VideoReader("video/test_video.mp4")
model = inference.init_detector("config.py", "work_dir/epoch_18.pth", "cuda:0")


fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
vWrite = cv.VideoWriter("work_dir/output.mp4", fourcc, video.fps, video.resolution, True)
for i in range(len(video)):
    bgr_img = video[i]
    try:
        gray_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)[:, :, None]
    except:
        continue
    gray_img = np.repeat(gray_img, 3, axis=-1)
    output = inference_detector(model, bgr_img)
    for picture in output[1][0]:
        gray_img[picture] = bgr_img[picture]

    model.show_result(
        gray_img,
        output[0],
        score_thr=.3,
        show=False,
        wait_time=0,
        win_name="show",
        bbox_color=None,
        text_color=(200, 200, 200),
        mask_color=None,
        out_file="cache.png"
    )
    vWrite.write(cv.imread("cache.png")[..., ::-1])

vWrite.release()
