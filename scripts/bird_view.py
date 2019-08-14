import cv2
import numpy as np

#Пока что всё как обычно открываем картинку, задаём найденные и целевые точки,
image = cv2.imread('38.png')

pts1 = np.float32([[422, 184], [455, 184], [358, 599], [523, 599]])
pts2 = np.float32([[0, 0], [150, 0], [0, 800], [150, 800]])
#print(pts2)
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(image, matrix, (450, 1400))


cols, rows, chan = image.shape
inputCorners = np.array([[0, 0], [rows, 0], [0, cols], [rows, cols]], dtype='float32')
inputCorners = np.array([inputCorners])
#print('положение углов исходной картинке, по ширине и высоте', inputCorners)

outputCorners = cv2.perspectiveTransform(inputCorners, matrix)
#rint('положение углов после трансформации:', outputCorners)

x,y,w,h = cv2.boundingRect(outputCorners)
x1,y1,w1,h1 = cv2.boundingRect(inputCorners)
#print("ограничительнная рамка координаты:", x,y,w,h)

for i in range(4):
  pts2[i]+= (-x, y)
#print('сдвиг целевых точек (новая оргничительная рамка):', pts2)

M = cv2.getPerspectiveTransform(pts1, pts2);
# Применяем трансформацию к картинке, размер - как ограничительная рамка
output = cv2.warpPerspective(image, M, (w, h))
print(output.shape)
output = cv2.pyrDown(output)
cv2.imshow("Frame", image)
cv2.imshow("Perspective transformation", result)
cv2.imshow("NEW Perspective transformation", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

