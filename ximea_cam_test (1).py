from ximea import xiapi
import cv2
import numpy as np
import matplotlib.pyplot as plt
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
img4 = cv2.imread("save/img3.jpg")



zoz = []
while cv2.waitKey(1) != ord('q'):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image,(1024,1024))
    cv2.imshow("test2", image)

    if  cv2.waitKey(1) == ord('h'):
        cv2.imwrite(f"img/img{image_num}.jpg", image)
        image_num += 1



    if cv2.waitKey(1) == ord('p'):
        
        

        Hori = np.concatenate((a, b), axis=1) 
        Hori2 = np.concatenate((c, img4), axis=1) 
        all = np.concatenate((Hori, Hori2), axis=0)
        all = cv2.resize(all, (512, 512))

        cv2.imshow('ALL', all) 
        


    if cv2.waitKey(1) == ord('k'):
        print("k")
        Identity = np.array([[1,1,1],[1,0,1],[1,1,1]])
        a = cv2.filter2D(a,-1,Identity)
    


    if cv2.waitKey(1) == ord('u'):
        img4 = cv2.imread("C:/Users/legi1/Documents/GitHub/Ximea_camera/save/img3.jpg")
        rotated = img4.copy
        print("start")

        x, y, _ = img4.shape

        for i in range(x):
            for j in range(y):
                rotated[j,i,:] = img4[i,j, :]

        print("end")
        img4 = rotated
        cv2.waitKey(0)


    if cv2.waitKey(1) == ord("r"):
        img3 = c
        red = np.array(img3)
        print("start")


        red[:, :, 0] = 0
        red[:, :, 1] = 0


        print("end")
        c = red
        cv2.waitKey(0)
                
 
 

 
 
#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()
 
#stop communication
cam.close_device()
cv2.destroyAllWindows()
print('Done.')