import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from Hough import *

focal_dist = 0.012   # фокусное расстояние камеры (м.)
rail_height = 0.25   # высота рельса над уровнем земли (м.)
distrail_m = 3.3   # расстояние до 6-ой шпалы отн-но земли без учета мертвой зоны (м.)
# 88 пикселей = 6 шпал = 3.3 метра (примерно)
questioned_obj_dist_p = 88   # расстояние до искомого объекта (в пикселях)
questioned_obj_height_p = 125 #70   # высота человека в пикселях (= высота bounding box)
questioned_obj_height_m = 1.62393 #1.8   # предполагаем, что рост человека = 1,8 м.
dead_zone_m = 13   # расстояние мертвой зоны (м.)




questioned_obj_dist_m = ((distrail_m / rail_height) * questioned_obj_height_p) * questioned_obj_height_m / questioned_obj_height_p
questioned_obj_dist_m += dead_zone_m
print('РАССТОЯНИЕ ДО ЧЕЛОВЕКА (метров): ', questioned_obj_dist_m)

def get_bb_coords():
	# Допустим, у нас имеется bounding box:
	# Это на картинке 3.jpeg человек слева от рельс (второй)
	tl = (300, 335)
	tr = (350, 335)
	bl = (300, 460)
	br = (350, 460)

	# Это на картинке 2.jpeg человек на рельсах
	#tl = (431, 230)
	#tr = (462, 230)
	#bl = (431, 310)
	#br = (462, 310)
	return (tl, tr, bl, br)



def get_camera_displacement(img_path):
	"""Находим смещение камеры относительно центра"""
	
	rails = make_Hough_transform(img_path)
	image = cv2.imread(img_path)
	image_width_center = int(image.shape[1] // 2)   # центр кадра по длине
	#print('Середина кадра: {}-й пиксель\n'.format(image_width_center))
	rail1 = rails[0][0]   # список координат первой рельс
	rail2 = rails[1][0]
	ind1 = list(rail1).index(max(rail1))
	ind2 = list(rail2).index(max(rail2))
	if ind1 <= 1:
		rail1_w = rail1[ind1-1]
		rail2_w = rail2[ind2-1]
	else:
		rail1_w = rail1[ind1+1]
		rail2_w = rail2[ind2+1]

	#print('Координата левой рельсы: {} px.\n'.format(rail1_w))
	#print('Координата правой рельсы: {} px.\n'.format(rail2_w))
	rails_between_m = abs(rail1_w-rail2_w)
	#print('Расстояние между рельсами: {} px.\n'.format(rails_between_m))
	midpoint_rails = int((rail1_w+rail2_w) // 2)   # серединная точка на рельсах
	#print('Серединная точка на рельсах: {} px.\n'.format(midpoint_rails))
	displacement = abs(image_width_center - midpoint_rails)
	#print('Смещение камеры: {} px.'.format(displacement))
	midpoint_rails = (midpoint_rails, image.shape[0])
	return midpoint_rails, displacement, rails_between_m


def get_bottom_midpoint():
	"""Находим координаты середины нижней стороны bbox"""

	tl, tr, bl, br = get_bb_coords()
	return ((bl[0] + br[0]) // 2, bl[1])


def get_questioned_obj_dist_m(img_path, dists, coeffs, dead_zone_m=13):
	"""Вычисляем расстояние между объектом и камерой"""

	image = cv2.imread(img_path)
	tl, tr, bl, br = get_bb_coords()
	man_ratio = abs(tr[1]-br[1]) / abs(tl[0]-tr[0])   # во сколько раз человек по высоте больше, чем по ширине
	print(man_ratio)
	dist_x_proj, dist_rails_proj = get_questioned_obj_dist_p_axis(img_path)
	dists = [int(dist) for dist in dists]
	num = list(dists).index(dist_x_proj)
	

	bb_width_p = abs(tr[0]-tl[0])
	bb_height_p = abs(tl[1]-bl[1])
	obj_width_m = (1520*bb_width_p / dist_rails_proj) / 10
	obj_height_m = obj_width_m * man_ratio
	#print('Ширина объекта = {:.3f} см.'.format(obj_width_m))
	#print('Высота объекта = {:.3f} см.'.format(obj_height_m))

	# ----------------------------------------------------

	coeffs = 1/np.array(coeffs)   # сколько содержится единиц информации в одном пикселе
	dist_x_proj_p_scaled = sum(coeffs[int(-dist_x_proj):-1])   # ищем количество пикселей, которое занимает расстояние от объекта до 0-й строки, если бы не было бы кменьшения масштаба каждой из строк
	
	print((bl[0] + br[0])//2)   # расстояние от начала изображения до середины bb (4)
	_, _, rails_between_p = get_camera_displacement(img_path)   # расстояние между рельсами
	dist_x_proj_m_scaled = (dist_x_proj_p_scaled * 1520 / rails_between_p) / 1000
	print('Расстояние между объектом и осью x при опускании перпендикуляра (метров): ', dist_x_proj_m_scaled)
	dist_obj_railmid_m = ((dist_rails_proj * 1520) / rails_between_p) / 1000
	print('Расстояние между объектом и центром рельс (проекция на ось x, в метрах): ', dist_obj_railmid_m)
	dist_obj_m = np.sqrt(dist_x_proj_m_scaled ** 2 + dist_obj_railmid_m ** 2)
	dist_obj_m += dead_zone_m
	print('РАССТОЯНИЕ ДО ОБЪЕКТА (в метрах): ', dist_obj_m)






def get_questioned_obj_dist_p_axis(img_path, show_projections=True):
	"""Вычисляем расстояние от нижней стороны bbox до оси x и середины рельс в пикселях"""

	image = cv2.imread(img_path)
	bb_bottom_midpoint = get_bottom_midpoint()
	rails = make_Hough_transform(img_path)
	midpoint_rails, _, _ = get_camera_displacement(img_path)

	"""Расстояние от нижней части bbox до оси x"""
	
	x_proj = (bb_bottom_midpoint[0], image.shape[0])   # Координаты проекции по оси x
	dist_x_proj = distance.euclidean(x_proj, bb_bottom_midpoint)
	print('Расстояние по x: {} px.'.format(int(dist_x_proj)))
	
	rails_proj = (midpoint_rails[0], bb_bottom_midpoint[1])   # Координаты проекции на рельсы
	dist_rails_proj = distance.euclidean(rails_proj, bb_bottom_midpoint)
	print('Расстояние до середины рельс: {} px.'.format(int(dist_rails_proj)))

	if show_projections:
		proj_image = np.copy(image) * 0  # creating a blank to draw lines on
		cv2.line(proj_image, x_proj, bb_bottom_midpoint, (0,0,250), 3)
		cv2.line(proj_image, rails_proj, bb_bottom_midpoint, (0,0,250), 3)
		# Draw the lines on the image
		lines_proj = cv2.addWeighted(image, 0.8, proj_image, 1, 0)
		
		# Середина проекции на ось x. Расположим тут надпись о расстоянии в px до оси x.
		mid_proj_x = (bb_bottom_midpoint[0], (int(bb_bottom_midpoint[1] + x_proj[1])//2))

		# Середина прокуции на рельсы. 
		mid_rails_proj = ((rails_proj[0]+bb_bottom_midpoint[0])//2, bb_bottom_midpoint[1]-10)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(lines_proj, '{}px'.format(int(dist_x_proj)), mid_proj_x, font, 0.85, (0,0,250), 2, cv2.LINE_AA)
		cv2.putText(lines_proj, '{}px'.format(int(dist_rails_proj)), mid_rails_proj, font, 0.85, (0,0,250), 2, cv2.LINE_AA)
		cv2.imshow('Projection', lines_proj)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	
	return dist_x_proj, dist_rails_proj



def get_dists_between_rails(lines):
	"""Получаем попиксельные расстояния между рельсами"""

	dists = []
	coords_rail1 = get_intermediate_line_values(lines[0])
	coords_rail2 = get_intermediate_line_values(lines[1])
	rails_pair_coords = list(zip(coords_rail1, coords_rail2))
	for pair in rails_pair_coords:
		dist = distance.euclidean(pair[0], pair[1])
		dists.append(dist)
	return dists


def get_scale_coeffs(img_path):
	"""Получаем масштабирующий коэффициент на каждую строчку матрицы"""

	lines = make_Hough_transform(img_path)
	dists = get_dists_between_rails(lines)

	coeffs = []
	for i in range(1, len(dists)):
		coeff = dists[0] / dists[i]
		coeffs.append(coeff)
	return coeffs


def get_intermediate_line_values(line):
	"""Получаем промежуточные координаты для рельс (на каждый пиксель)"""

	x1 = line[0][0]
	y1 = line[0][1]
	x2 = line[0][2]
	y2 = line[0][3]
	k = (y2-y1) / (x2-x1)
	b = (y1*x2)/(x2-x1) - (x1*y2)/(x2-x1)

	if y2 < y1:
		y1, y2 = y2, y1
	xs = []
	for y in range(y1, y2):
		x = (y-b)/k
		xs.append(x)
	ys = [i for i in range(y1, y2)]
	return list(zip(xs, ys))






img_path = '3.jpeg'
lines = make_Hough_transform(img_path)
#get_camera_displacement(img_path)

dist_x_proj, _ = get_questioned_obj_dist_p_axis(img_path)

coeffs = get_scale_coeffs(img_path)
dists = get_dists_between_rails(lines)
get_questioned_obj_dist_m(img_path, dists, coeffs)
