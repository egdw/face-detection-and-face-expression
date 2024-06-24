"""
人脸识别加面部心情检测
"""
import base64
# load libraries
import time
from io import BytesIO

import PIL
import torch
from PIL.ImageDraw import ImageDraw
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image

# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
# Use a pipeline as a high-level helper
from transformers import pipeline


class FacialExpressionRecognition:
    def __init__(self, device="cpu"):
        self.device = device
        # 人脸检测模型初始化
        face_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        self.face_model = YOLO(face_model_path)

        self.express_pipe = pipeline("image-classification", model="motheecreator/vit-Facial-Expression-Recognition",
                                     device=self.device)
        # self.express_processor = AutoImageProcessor.from_pretrained("motheecreator/vit-Facial-Expression-Recognition")
        # self.express_model = AutoModelForImageClassification.from_pretrained(
        #     "motheecreator/vit-Facial-Expression-Recognition")
        self.img = None

    """
    将图片转换为base64格式
    image 表示等待转换的图片
    """
    def __image2base64__(self, image):
        buff = BytesIO()
        image.save(buff, format="JPEG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        return str(img_str)

    """
    img：代表传入的PIL图像
    result：表示人脸的识别结果
    """

    def draw_faces(self, img, result):
        if type(img) == str:
            img = Image.open(img)
            img = img.convert('RGB')
        draw: PIL.ImageDraw = ImageDraw(img)
        xyxy = result.xyxy
        confidence = result.confidence
        for xy, conf in zip(xyxy, confidence):
            # print(xy, conf)
            draw.rectangle(xy, outline="blue", width=2)
            # 计算标签位置
            text_x = xy[0] + 1  # 标签位置稍微偏移边界框
            if text_x < 0:
                text_x = 0
            if text_x > img.width:
                text_x = img.width
            text_y = xy[1] - 20 - 1
            if text_y < 0:
                text_y = 0
            if text_y > img.height:
                text_y = img.height
            # 绘制置信度标签
            draw.text((text_x, text_y), str(conf), fill='blue')

        img.save("result.jpg")
        return self.__image2base64__(img)

    """
    检测人脸
    """
    def detect_face(self, img):
        self.img = img
        # output = self.face_model(img, device=device, show=True, show_boxes=True)
        # 人脸识别
        output = self.face_model(img, device=self.device)
        # 获得人脸识别结果
        results = Detections.from_ultralytics(output[0])
        # 画出定位图
        faces_rect_img = self.draw_faces(img, results)  # 获得人脸定位图
        # 得到每个人脸的心情
        faces_dict = self.analyze_faces(results.xyxy)
        # 返回最终结果
        return {"faces": faces_rect_img, "data": faces_dict}

    """
    分析人脸表情
    """
    def analyze_faces(self, faces):
        if self.img is None:
            return
        faces_dict = []
        if type(self.img) == str:
            self.img = Image.open(self.img)
        image_face = self.img.convert('RGB')
        for i, face in enumerate(faces):
            cropped_face_image = image_face.crop(face)
            img_base64 = self.__image2base64__(cropped_face_image)
            # cropped_face_image.save('face{}.png'.format(i))
            result = self.express_pipe(images=[cropped_face_image])[0][0]
            result["xyxy"] = face.tolist()
            result["img"] = img_base64
            faces_dict.append(result)
        return faces_dict
