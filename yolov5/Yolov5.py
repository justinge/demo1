from random import random
import random
import numpy as np
import torch
import cv2
import yaml

from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


class YOLOv5(object):
    # 参数设置
    _defaults = {
        "weights": "yolov5s.pt",
        "imgsz": 640,
        "iou_thres": 0.45,
        "conf_thres": 0.25,
        "classes": 0  # 只检测人
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # 初始化操作，加载模型
    def __init__(self, device='0', **kwargs):
        self.__dict__.update(self._defaults)
        self.device = select_device(device)
        self.half = self.device != "cpu"
        self.model = attempt_load(self.weights, device=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16

    # 推理部分
    def infer(self, inImg):
        # 使用letterbox方法将图像大小调整为640大小
        img = letterbox(inImg, new_shape=self.imgsz)[0]

        # 归一化与张量转换
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 推理
        pred = self.model(img, augment=True)[0]
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=True)

        bbox_xyxy = []
        confs = []
        cls_ids = []

        # 解析检测结果
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # 将检测框映射到原始图像大小
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], inImg.shape).round()
                # 保存结果
                for *xyxy, conf, cls in reversed(det):
                    bbox_xyxy.append(xyxy)
                    confs.append(conf.item())
                    cls_ids.append(int(cls.item()))

                xyxys = torch.Tensor(bbox_xyxy)
                confss = torch.Tensor(confs)
                cls_ids = torch.Tensor(cls_ids)

        return xyxys, confss, cls_ids


# 将图像以jpg编码，并转换为字节流
def get_img_bytes(img):
    img_str = cv2.imencode('.jpg', img)[1].tobytes() if img is not None else None
    return img_str


# 定义工具方法，在原始图像上画框
def plot_one_box(x, img, color=None, label="person", line_thickness=None):
    """ 画框,引自 YoLov5 工程.
    参数:
        x:      框， [x1,y1,x2,y2]
        img:    opencv图像
        color:  设置矩形框的颜色, 比如 (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


if __name__ == '__main__':
    with open("data/coco128.yaml", 'r') as f:
        class_names = yaml.safe_load(f)["names"]
    yolov5_obj = YOLOv5()
    img = cv2.imread("data/images/zidane.jpg")  # image :是返回提取到的图片的值
    out = yolov5_obj.infer(img)
    boxs, confs, ids = out[0].tolist(), out[1].tolist(), out[2].tolist()
    ids = list(map(int, ids))
    print(ids)

    if boxs is not None:
        for i, box in enumerate(boxs):
            plot_one_box(box, img, label=str(class_names[ids[i]]))
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
        cv2.imwrite("first.png", img)
