import cv2
import numpy as np

image = cv2.imread('38.png')
#Задаём найденные и целевые точки соответственно
pts1 = np.float32([[422, 184], [455, 184], [358, 599], [523, 599]])
pts2 = np.float32([[0, 0], [150, 0], [0, 800], [150, 800]])
#Находим матрицу трансформации
matrix_1 = cv2.getPerspectiveTransform(pts1, pts2)
#Применяем трансформацию к картинке
result = cv2.warpPerspective(image, matrix_1, (150, 800))


cols, rows, chan = image.shape
#Выясняем положение координат углов исходной картинки
inputCorners = np.array([[0, 0], [rows, 0], [0, cols], [rows, cols]], dtype='float32')
inputCorners = np.array([inputCorners])
#Применяем трансформацию и получаем новые координаты углов
outputCorners = cv2.perspectiveTransform(inputCorners, matrix_1)
#Находим ограничительную рамку для данного изображения
x,y,w,h = cv2.boundingRect(outputCorners)
print('Ограничительная рамка (х, у, ш, длина, ширина):', x,y,w,h)

#Смещаем целевые точки в противоположное направление, от того, куда ушла рамка
for i in range(4):
  pts2[i] += (-x, y)

#Вычисляем новую матрицу трансформации для новых целевых точек
matrix_2 = cv2.getPerspectiveTransform(pts1, pts2);
#Применяем трансформацию к картинке, размер равен ограничительной рамке
output = cv2.warpPerspective(image, matrix_2, (w, h))


output = cv2.pyrDown(output)
cv2.imshow("Original", image)
cv2.imshow("Only rails", result)
cv2.imshow("Bird view", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

