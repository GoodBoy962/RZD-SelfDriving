import numpy as np
import cv2
from math import floor


def inv_perspective(image, camera_param):
    """ Inverse perspective Mapping """
    roi_left = -10
    roi_right = 10
    roi_near = get_interpolation(camera_param, image.shape[1])
    length = 100

    objpoint = np.array([(roi_left, 0, roi_near),
                         (roi_right, 0, roi_near),
                         (roi_left, 0, roi_near + length),
                         (roi_right, 0, roi_near + length)], dtype='double')

    imgpoint, _ = cv2.projectPoints(objpoint, camera_param[2], camera_param[3],
                                    camera_param[0], camera_param[1])

    pts1 = np.float32([
        [imgpoint[0][0][0], imgpoint[0][0][1]],
        [imgpoint[1][0][0], imgpoint[1][0][1]],
        [imgpoint[2][0][0], imgpoint[2][0][1]],
        [imgpoint[3][0][0], imgpoint[3][0][1]]])

    scale = 20    
    out_width = 2 * (scale * roi_right)
    out_height = scale * length

    pts2 = np.float32([
            [0, out_height],
            [out_width, out_height],
            [0, 0],
            [out_width, 0]])

    return cv2.getPerspectiveTransform(pts1, pts2)


def get_interpolation(camera_param, find_pix):
    """ Find the nearest point to train on the image """
    test_obj = []
    max_meter = 20
    min_meter = 3
    for i in range(min_meter, max_meter):
        test_obj.append((0, 0, i))

    test_point = np.array(test_obj, dtype='double')
    test_img, _ = cv2.projectPoints(test_point, camera_param[2], camera_param[3],
                                    camera_param[0], camera_param[1])
    p = []
    for t in test_img:
        if t[0][1] > 0:
            p.append(t[0][1])

    l = len(p) - 1
    lp = len(p)
    low = max_meter - lp
    m = 16
    met = 0

    while l > 0:
        if find_pix > floor(p[l]):
            l -= 1
        else:
            diff = floor(p[l]) - floor(p[l + 1])
            f = floor(p[l]) - find_pix
            met = f / diff
            ind = l
            m = ind + low
            break

    result = float('{0:.1f}'.format(m + met))

    return result
    
""" Convert pixel points from original image to local coordinates (meters) """
# points - array of points on the original image
pts = np.float32(points)
converted_pts = cv2.perspectiveTransform(pts, M)

m_coord = []
for pt in converted_pts:
	x, y = pt[0][0], pt[0][1]
    x, y = int(x), int(y)
    
    x = (x / out_width * (roi_right-roi_left) + roi_left)*-1
    y = (out_height - y) / out_height * length + roi_near
    
    m_coord.append((x, y))

result = m_coord
    