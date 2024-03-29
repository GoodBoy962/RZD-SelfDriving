import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import imageai
from imageai.Detection import VideoObjectDetection
import os


def detect_with_mrcnn():

    # Root directory of the project
    ROOT_DIR = os.path.abspath("../")

    # Import Mask RCNN
    sys.path.append(ROOT_DIR)  # To find local version of the library
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    # Import COCO config
    sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
    import coco

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    class InferenceConfig(coco.CocoConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()


    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    class_names = [
        'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]


    def random_colors(N):
        np.random.seed(1)
        colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
        return colors


    colors = random_colors(len(class_names))
    class_dict = {name: color for name, color in zip(class_names, colors)}


    def apply_mask(image, mask, color, alpha=0.5):
        """apply mask to image"""
        for n, c in enumerate(color):
            image[:, :, n] = np.where(
                mask == 1,
                image[:, :, n] * (1 - alpha) + alpha * c,
                image[:, :, n]
            )
        return image


    def display_instances(image, boxes, masks, ids, names, scores):
        """take the image and results and apply the mask, box, and Label"""
        n_instances = boxes.shape[0]

        if not n_instances:
            print('NO INSTANCES TO DISPLAY')
        else:
            assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

        for i in range(n_instances):
            if not np.any(boxes[i]):
                continue

            y1, x1, y2, x2 = boxes[i]
            label = names[ids[i]]
            color = class_dict[label]
            score = scores[i] if scores is not None else None
            caption = '{} {:.2f}'.format(label, score) if score else label
            mask = masks[:, :, i]

            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        return image


        
    capture = cv2.VideoCapture(0)

    # these 2 lines can be removed if you dont have a 1080p camera.
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = capture.read()
        #frame = cv2.resize(frame, (90, 90))
            
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()







def detect_with_imageai_yolov3():

    """
    <-- CODE IF YOU NEED TO LOAD SOME VIDEO -->
    execution_path = os.getcwd()
    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "weights/yolo.h5"))
    detector.loadModel()

    video_path = detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, "example_video.mp4"),
        output_file_path=os.path.join(execution_path, "Yolo.mp4"),
        frames_per_second=20, log_progress=True
    )
    """

    execution_path = os.getcwd()
    camera = cv2.VideoCapture(0)
    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "weights/yolo.h5"))
    detector.loadModel()


    video_model = detector.detectObjectsFromVideo(camera_input=camera,
                                                  output_file_path=os.path.join(execution_path, "camera_detected_video"),
                                                  frames_per_second=20,
                                                  log_progress=True,
                                                  minimum_percentage_probability=40)



def detect_with_imageai_yolotiny():
    
    execution_path = os.getcwd()
    camera = cv2.VideoCapture(0)
    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(execution_path, "weights/yolo_tiny.h5"))
    detector.loadModel()


    video_model = detector.detectObjectsFromVideo(camera_input=camera,
                                                  output_file_path=os.path.join(execution_path, "camera_detected_video"),
                                                  frames_per_second=20,
                                                  log_progress=True,
                                                  minimum_percentage_probability=40)




def detect_with_imageai_retinanet():

    def forFrame(frame_number, output_array, output_count):
        """Детекция bounding box'ов"""

        print("ДЛЯ КАДРА " , frame_number)
        print('Объект:', output_array[0]['name'])
        print('Вероятность:', output_array[0]['percentage_probability'])
        print('Bounding box:', output_array[0]['box_points'])
        print("Уникальных объектов: ", output_count[output_array[0]['name']])
        print("------------END OF A FRAME --------------\n\n")


    execution_path = os.getcwd()
    camera = cv2.VideoCapture(0)
    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path, "weights/resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()


    video_model = detector.detectObjectsFromVideo(camera_input=camera,
                                                  output_file_path=os.path.join(execution_path, "camera_detected_video"),
                                                  frames_per_second=20,
						  per_frame_function=forFrame,
                                                  minimum_percentage_probability=40)



#detect_with_imageai_mrcnn()
#detect_with_imageai_yolov3()
#detect_with_imageai_yolotiny()
detect_with_imageai_retinanet()
