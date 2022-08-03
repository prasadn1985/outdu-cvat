import json
import base64
import io
from PIL import Image

import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from skimage.measure import approximate_polygon, find_contours


# import PointRend project
from detectron2.projects import point_rend

from detectron2.data.detection_utils import convert_PIL_to_numpy
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
import numpy as np



CONFIG_OPTS = ["MODEL.WEIGHTS", "model_final_ba17b9.pkl"]
CONFIDENCE_THRESHOLD = 0.5

def init_context(context):
    context.logger.info("Init context...  0%")

    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file("detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml")
    if torch.cuda.is_available():
        CONFIG_OPTS.extend(['MODEL.DEVICE', 'cuda'])
    else:
        CONFIG_OPTS.extend(['MODEL.DEVICE', 'cpu'])

    cfg.merge_from_list(CONFIG_OPTS)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    context.user_data.model_handler = predictor

    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run pointrend-X101-FPN model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = convert_PIL_to_numpy(Image.open(buf), format="BGR")

    predictions = context.user_data.model_handler(image)

    instances = predictions['instances']
    pred_boxes = instances.pred_boxes
    scores = instances.scores
    pred_classes = instances.pred_classes
    masks = instances.pred_masks
    results = []
    for i in range(len(pred_boxes)):
        score = scores[i]
        label = COCO_CATEGORIES[int(pred_classes[i])]["name"]
        mask = masks[i]
        if score >= threshold:
            print(mask)
            if torch.cuda.is_available():
                mask=mask.cpu()
            mask = np.array(mask, dtype=np.uint8)
            contours = find_contours(mask, CONFIDENCE_THRESHOLD)
            # only one contour exist in our case
            contour = contours[0]
            contour = np.flip(contour, axis=1)
            # Approximate the contour and reduce the number of points
            contour = approximate_polygon(contour, tolerance=2.5)
            if len(contour) < 6:
                continue

            results.append({
                "confidence": str(score),
                "label": label,
                "points": contour.ravel().tolist(),
                "type": "polygon",
            })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)
