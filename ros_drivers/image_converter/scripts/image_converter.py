#!/usr/bin/env python
import rospy
import sys
import os
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageConverter:

    def __init__(self, topic):
        self.bridge = CvBridge()
        self.subscriber = rospy.Subscriber(topic, Image, self.callback)

    def callback(self, data):
        '''
            callback for videos stream
            get camera output and apply...
        '''
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)

        cv2.imshow('Camera output', cv_image)
        cv2.waitKey(3)


def main(args):
    ic = ImageConverter('pylon_camera_node/image_raw')
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.lofinfo('Shutting down')
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)

