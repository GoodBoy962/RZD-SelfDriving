import xml.etree.ElementTree as ET
import xmltodict
from imageai.Detection import ObjectDetection
import os
from scipy.spatial import distance
import numpy as np


def get_bbox_yolo(path_to_images, path_to_weights='/home/user/aylifind/weights/yolo.h5', output_folder='/home/user/aylifind/output/'):
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(path_to_weights)
    detector.loadModel()
    for curr_image in os.listdir(path_to_images):
        bboxes_per_image = []
        image_path = path_to_images + curr_image
        custom = detector.CustomObjects(person=True)
        detections = detector.detectCustomObjectsFromImage(input_image=image_path,
                                                           output_image_path=output_folder,
                                                           custom_objects=custom,
                                                           minimum_percentage_probability=30)
        for eachObject in detections:
            try:
                x1 = int(eachObject['box_points'][0])
                y1 = int(eachObject['box_points'][1])
                x2 = int(eachObject['box_points'][2])
                y2 = int(eachObject['box_points'][3])
                bboxes_per_image.append((x1, y1, x2, y2))
            except:
                continue
        return bboxes_per_image



def parse_xml(xml_path):
    with open(xml_path) as fd:
        doc = xmltodict.parse(fd.read())
    bbox_num = len(doc['annotation']['object']) # количество bbox'ов на картинке
    bboxes_per_image = []
    for i in range(0, bbox_num):
        bbox = doc['annotation']['object'][i]
        xmin = int(bbox['bndbox']['xmin'])
        ymin = int(bbox['bndbox']['ymin'])
        xmax = int(bbox['bndbox']['xmax'])
        ymax = int(bbox['bndbox']['ymax'])
        bboxes_per_image.append((xmin, ymin, xmax, ymax))
    return bboxes_per_image



bbox_yolo = get_bbox_yolo('/home/user/aylifind/images/')
bbox_expert = parse_xml('/home/user/aylifind/train/1.xml')


if len(bbox_yolo) < len(bbox_expert):
    bbox_yolo.append((0,0,0,0))


    
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle



    xA = max(boxA[0], boxB[0])
    print(xA)
    yA = max(boxA[1], boxB[1])
    print(yA)
    xB = min(boxA[2], boxB[2])
    print(xB)
    yB = min(boxA[3], boxB[3])
    print(yB)

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


iou = bb_intersection_over_union(bbox_yolo[0], bbox_expert[1])
print(iou)

"""
# Попытки подогнать наиболее подходящие bb так, чтобы они стояли попарно
print('bbox_yolo: ', bbox_yolo)
print('bbox_expert: ', bbox_expert)

dsts = []
i, j = 0, 0
for bb_yolo in bbox_yolo:
    i += 1   # количество задетектинных yolo людей
    for bb_expert in bbox_expert:
        dst = distance.euclidean(bb_yolo, bb_expert)
        dsts.append(dst)

dsts = np.array(sorted(dsts))
dsts = dsts.reshape(i, int(len(dsts)/i))



#print(dsts.reshape(len(bb_yolo), len(bb_expert)))

#min(len(bbox_yolo), len(bbox_expert))
"""




