import torch
import numpy as np
import cv2
from models.experimental import attempt_load
from utils.general import check_img_size,  non_max_suppression, scale_coords
from utils.datasets import letterbox

weights="yolov7.pt"
IMG_SZ=640

class ModelHandler:
    def __init__(self, labels):
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        self.labels=labels

        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(IMG_SZ, s=self.stride)  # check img_size
        self.model.half()


    def infer(self, image, threshold):
        image = np.array(image)
        # Padded resize
        img = letterbox(image, self.imgsz, stride=self.stride)[0]

        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Run inference
        self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once 

        img = torch.from_numpy(img).to(self.device)
        img = img.half() #if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, threshold, 0.45)
        print("----pred--")
        print(pred)
        print("--pred----")

        results = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                print(det)
                for *xyxy, conf, cls in reversed(det):
                    obj_class = int(cls.item())
                    print("--xyxy")
                    print(xyxy)
                    print("--xyxy---")

                    results.append({
                                        "confidence": str(conf.item()),
                                        "label": self.labels.get(obj_class, "unknown"),
                                        "points": [xyxy[0].item(),xyxy[1].item(),xyxy[2].item(),xyxy[3].item()],
                                        "type": "rectangle",
                                    })
                    print(results)

        return results