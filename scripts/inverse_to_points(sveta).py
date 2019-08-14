import numpy as np
import cv2
from math import floor

mtxL = [3.100083007812500000e+03,0.000000000000000000e+00,1.024000000000000000e+03,0.000000000000000000e+00,3.100083007812500000e+03,7.680000000000000000e+02, 0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00]
distL = [0, 0, 0, 0, 0]
rvec = [0.156107,  0.035085,  0.006041]
tvec = [-0.016078,  4.922029, -1.911572]
camera_param = [np.array(mtxL, dtype='double').reshape(3, 3), np.float32(distL).reshape(1, 5), np.array(rvec), np.array(tvec)]
#print(camera_param)
#camera_param=([(3.100083007812500000e+03,0.000000000000000000e+00,1.024000000000000000e+03,0.000000000000000000e+00,3.100083007812500000e+03,7.680000000000000000e+02, 0.000000000000000000e+00,0.000000000000000000e+00,1.000000000000000000e+00), (0, 0, 0, 0, 0), (0.156107,  0.035085,  0.006041), (-0.016078,  4.922029, -1.911572)])

image = cv2.imread('3_.jpeg')
image=cv2.resize(image, (2064, 1544))
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
    find_pix = 6020
    test_obj = []
    max_meter =20
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
            diff = abs(floor(p[l]) - floor(p[l + 1]))
            f = floor(p[l]) - find_pix
            met = f / diff
            ind = l
            m = ind + low
            break

    result = float('{0:.1f}'.format(m + met))

    return result
    
""" Convert pixel points from original image to local coordinates (meters) """
# points - array of points on the original image
M = inv_perspective(image, camera_param)
print('матрица трансформации', M)


roi_left = -10
roi_right = 10
length = 100
m_coord = []
scale = 20    
out_width = 2 * (scale * roi_right)
out_height = scale * length
roi_near = get_interpolation(camera_param, image.shape[1])


pts = np.array([[591, 588]], dtype='float32')
pts = np.array([pts])
converted_pts = cv2.perspectiveTransform(pts, M)
out_img = cv2.warpPerspective(image, M, (out_width, out_height))

for pt in converted_pts:
    x, y = pt[0][0], pt[0][1]
    x, y = int(x), int(y)
    x = (x / out_width * (roi_right-roi_left) + roi_left)*-1
    y = (out_height - y) / out_height * length + roi_near
    m_coord.append((x, y))
    result = m_coord
print('координаты', result) 

out_img = cv2.pyrDown(out_img)
cv2.imshow("Frame", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
