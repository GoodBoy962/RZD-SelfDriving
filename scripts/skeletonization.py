import cv2
import numpy as np
from scipy import ndimage
import scipy
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from inverse_to_points_sveta import *
from sklearn import linear_model



img = cv2.imread('17m.png',0)


def make_skeletonization(img, show_image=False):
    kernel_size=5
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while (not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    skel += img
    if show_image:
        cv2.imshow("skel",skel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return skel


def make_Sobel_filtration(img, show_image=False):
    im = img.astype('int32')
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    mag = mag.astype(np.uint8)
    if show_image:
        cv2.imshow('Sobel', mag)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return mag


def make_Canny_filtration(img, show_image=False):
    im = img.astype('int32')
    edges = cv2.Canny(img, 100, 200)
    if show_image:
        cv2.imshow('Canny', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return edges


def convert_mask_to_regression(skel, show_image=False):
    """Перевод маски из 0 и 1 на изображении в регрессию на scatter plot"""

    indexes = list(zip(*np.where(skel > 0)))   # вывести все индексы, где яркость выше 0
    xs = [ind_pair[0] for ind_pair in indexes]
    ys = [ind_pair[1] for ind_pair in indexes]
    if show_image:
        plt.scatter(np.array(xs), np.array(ys))
        plt.show()
    return xs, ys


def cluster_mask_dbscan(skel, show_image=False):

    xs, ys = convert_mask_to_regression(skel)
    X = np.array(list(zip(xs, ys)))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=0.38)
    clusters = dbscan.fit_predict(X_scaled)
    if show_image:
        plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap="plasma")
        plt.title('DBScan')
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.grid()
        plt.show()
    return X[:, 0], X[:, 1]


def cluster_mask_kmeans(skel, show_image=False):

    xs, ys = convert_mask_to_regression(skel)
    X = np.array(list(zip(xs, ys)))

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    y_kmeans = kmeans.predict(X)

    if show_image:
        plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
        plt.xlabel('Feature 0')
        plt.ylabel('Feature 1')
        plt.title('KMeans')
        plt.grid()
        plt.show()


def make_regression(xs, ys):
    pass
    # Нужно сделать пары чисел (x, y) для каждого из кластеров
    #reg = linear_model.Ridge(alpha=.5)
    #reg.fit(left_rail)


if __name__ == '__main__':
    birdview = convert_to_points(img)
    skeleton = make_skeletonization(birdview)
    Sobel = make_Sobel_filtration(birdview)
    Canny = make_Canny_filtration(birdview)
    skeleton_Sobel = skeleton + Sobel
    skeleton_Canny = skeleton + Canny 

    cv2.imshow('Skeleton + Sobel', skeleton_Sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


    #cluster_mask_kmeans(skeleton, show_image=True)
    xs_cl, ys_cl = cluster_mask_dbscan(skeleton, show_image=True)
    make_regression(xs_cl, ys_cl)
