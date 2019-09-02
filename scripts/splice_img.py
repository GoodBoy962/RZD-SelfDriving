import cv2
import os


h = 2168
w = 4096

    
def make_video(path_to_img, path_to_vid, out_name='video_2', fps=2.0):
    
    data_names = sorted(os.listdir(path_to_img))

    #print(data_names[1])
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(path_to_vid+out_name+'.avi', fourcc, fps, (w, h), isColor=True) 
    for c in data_names:
        img = cv2.imread(path_to_img + c)
        video.write(img)    
    cv2.destroyAllWindows()
    video.release()



path_1 = '/home/user/Загрузки/MCC/img/'
path_2 = '/home/user/Загрузки/MCC/'
make_video(path_1, path_2)