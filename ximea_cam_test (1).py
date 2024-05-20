from ximea import xiapi
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###
 
#create instance for first connected camera
cam = xiapi.Camera()
 
 
 
#start communication
#to open specific device, use:
#cam.open_device_by_SN('41305651')
#(open by serial number)
print('Opening first camera...')
cam.open_device()
 
#settings
cam.set_exposure(10000)
cam.set_param('imgdataformat','XI_RGB32')
cam.set_param('auto_wb', 1)
print('Exposure was set to %i us' %cam.get_exposure())
 
#create instance of Image to store image data and metadata
img = xiapi.Image()
 
#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()
 
image_num = 0

a = cv2.imread("save/img0.jpg")
b = cv2.imread("save/img1.jpg")
c = cv2.imread("save/img2.jpg")
d = cv2.imread("save/img3.jpg")

def show(a, b, c, d):
    Hori = np.concatenate((a, b), axis=1) 
    Hori2 = np.concatenate((c, d), axis=1) 
    all = np.concatenate((Hori, Hori2), axis=0)
    all = cv2.resize(all, (512, 512))
    cv2.imwrite(f"save/all.jpg", all)


    print(f"dimesions: {all.shape}\ntype: {all.dtype}\nsize: {os.stat('save/all.jpg').st_size} bytes")
    cv2.imshow('ALL', all)
    
def kernel(a):
    print("kernel")
    Identity = np.array([[1,1,1],[1,0,1],[1,1,1]])
    a = cv2.filter2D(a,-1,Identity)
    return a

def rotate(d):
    img4 = d
    rotated = img4.copy()
    print("start")

    x, y, _ = d.shape

    for i in range(x):
        for j in range(y):
            rotated[j,i,:] = img4[i,j, :]

    print("end")
    d = rotated
    return d

def red(c):
    img3 = c
    red = np.array(img3)
    print("start")


    red[:, :, 0] = 0
    red[:, :, 1] = 0


    print("end")
    c = red
    return c


while cv2.waitKey(1) != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(1024,1024))
    cv2.imshow("test2", image)

    if  cv2.waitKey(1) == ord('h'):
        cv2.imwrite(f"img/img{image_num}.jpg", image)
        image_num += 1



    if cv2.waitKey(1) == ord('p'):
        show(a, b, c, d)


    if cv2.waitKey(1) == ord('k'):
        a = kernel(a)
    


    if cv2.waitKey(1) == ord('u'):
        d = rotate(d)


    if cv2.waitKey(1) == ord("r"):
        c = red(c)

    if cv2.waitKey(1) == ord("a"):
        print("start")
        a = kernel(a)
        print("kernel-----------------------------------------------------------------")
        d = rotate(d)
        print("rotate-----------------------------------------------------------------")
        c = red(c)
        print("red-----------------------------------------------------------------")
        show(a, b, c, d)
                
 
 
        
 
 
#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()
 
#stop communication
cam.close_device()
cv2.destroyAllWindows()
print('Done.')