# MaskRCNN launch

0. [optional] Run the virtual environment:

   `conda create -n MaskRCNN python=3.6 pip`

   `activate MaskRCNN`

1. Install packages from requirements. If you don't have GPU, you should replace `tensorflow-gpu==1.5` with `tensorflow==1.5`:

   `pip install -r requirements.txt`

2. Clone the repository:

   `git clone https://github.com/matterport/Mask_RCNN.git`

3. Clone one more repository:

   `git clone https://github.com/philferriere/cocoapi.git`

4. Use pip to install pycocotools:

   `pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

5. [optional] Install Visual C++ 2015 Build Tools

6. Download pretrained weights file `mask_rcnn_coco.h5` from

   **github.com/matterport/Mask_RCNN/releases** and place it to the `Mask_RCNN` directory.

7. To check all is fine, you may run `demo.ipynb` from `samples` directory.

8. To process video, launch `video_proc.ipynb`.


# ImageAI

ImageAI provides several models: Yolo, Yolo_Tiny and RetinaNet.

To install imageai, you must have installed dependencies specified at https://github.com/OlafenwaMoses/ImageAI#installation

You can download example video, results and weights for the models from **https://**















