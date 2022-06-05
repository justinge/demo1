# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run a Flask REST API exposing a YOLOv5s model
"""
import skimage
import urllib
import argparse
import io
import time

import numpy as np
import cv2
import torch
import yaml
from flask import Flask, request, Response
from PIL import Image

from flask import jsonify

from Yolov5 import YOLOv5, plot_one_box

app = Flask(__name__, static_folder="data/images")


@app.route("/image")
def index():
    with open("data/images/zidane.jpg", "rb") as f:
        resp = Response(f.read(), mimetype="image/jpeg")
    return resp


@app.route("/v1/img_url", methods=["POST"])
def img_url():
    import requests as req
    from PIL import Image
    from io import BytesIO
    json_data = request.get_json()
    response = req.get(json_data["img_url"])
    image = np.asarray(bytearray(response.content), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    start = time.time()
    out = yolov5_obj.infer(img)

    boxs, confs, ids = out[0].tolist(), out[1].tolist(), out[2].tolist()
    ids = list(map(int, ids))

    if boxs is not None:
        for i, box in enumerate(boxs):
            plot_one_box(box, img, label=str(class_names[ids[i]]))
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
        cv2.imwrite("first111.png", img)

    return jsonify(
        {"status": 0, "msg": "ok",
         "data": {"boxs": boxs, "confs": confs, "ids": ids, "result": "./fist.png",
                  "duration": time.time() - start}})


# 读取网络图片方法2
@app.route("/v1/img_url2", methods=["POST"])
def img_url2():
    json_data = request.get_json()
    img = skimage.io.imread(json_data["img_url"])

    start = time.time()
    out = yolov5_obj.infer(img)

    boxs, confs, ids = out[0].tolist(), out[1].tolist(), out[2].tolist()
    ids = list(map(int, ids))
    # 传入进去的还需要进行通道转换 @todo, 但看起来貌似问题不大
    if boxs is not None:
        for i, box in enumerate(boxs):
            plot_one_box(box, img, label=str(class_names[ids[i]]))
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
        skimage.io.imsave("first.png", img)

    return jsonify(
        {"status": 0, "msg": "ok",
         "data": {"boxs": boxs, "confs": confs, "ids": ids, "result": "./fist.png",
                  "duration": time.time() - start}})


# 读取网络图片方法1
@app.route("/v1/img_url1", methods=["POST"])
def img_url1():
    json_data = request.get_json()
    resp = urllib.request.urlopen(json_data["img_url"])
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    start = time.time()
    out = yolov5_obj.infer(img)

    boxs, confs, ids = out[0].tolist(), out[1].tolist(), out[2].tolist()
    ids = list(map(int, ids))

    if boxs is not None:
        for i, box in enumerate(boxs):
            plot_one_box(box, img, label=str(class_names[ids[i]]))
            # cv2.imshow("image", img)
            # cv2.waitKey(0)
        cv2.imwrite("first.png", img)

    return jsonify(
        {"status": 0, "msg": "ok",
         "data": {"boxs": boxs, "confs": confs, "ids": ids, "result": "./fist.png",
                  "duration": time.time() - start}})


DETECTION_URL = "/v1/object-detection/yolov5s"


class DealWithInterProcess:
    def process(self, yolov5_obj, img, file_name):
        # 执行推理
        start = time.time()
        out = yolov5_obj.infer(img)

        boxs, confs, ids = out[0].tolist(), out[1].tolist(), out[2].tolist()
        ids = list(map(int, ids))

        if boxs is not None:
            for i, box in enumerate(boxs):
                plot_one_box(box, img, label=str(class_names[ids[i]]))
                # cv2.imshow("image", img)
                # cv2.waitKey(0)
            cv2.imwrite(file_name, img)
        duration = time.time() - start
        return boxs, confs, ids, duration


@app.route(DETECTION_URL, methods=["POST"])
def predict():
    if request.method != "POST":
        return ""

    if request.files.get("image"):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))
        # # Method 2
        input_image = request.files["image"].read()
        imBytes = np.frombuffer(input_image, np.uint8)
        img = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)
        # img = cv2.imread("data/images/zidane.jpg")  # image :是返回提取到的图片的值
        file_name = "./file_name.png"
        boxs, confs, ids, duration = DealWithInterProcess.process(yolov5_obj, img, file_name)
        return jsonify(
            {"status": 0, "msg": "ok",
             "data": {"boxs": boxs, "confs": confs, "ids": ids, "result": file_name,
                      "duration": duration}})


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # opt = parser.parse_args()
    #
    # # Fix known issue urllib.error.HTTPError 403: rate limit exceeded https://github.com/ultralytics/yolov5/pull/7210
    # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    with open("data/coco128.yaml", 'r') as f:
        class_names = yaml.safe_load(f)["names"]
    yolov5_obj = YOLOv5()

    app.run(host="0.0.0.0", port=5000)  # debug=True causes Restarting with stat
