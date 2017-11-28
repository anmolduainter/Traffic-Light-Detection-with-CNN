from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
from keras.preprocessing import image
import pylab
import imageio
import cv2
import visvis as vv
from keras.models import load_model
from random import randint
import time




classifier = load_model('model.h5') 
                         
                   
def predict(test_image):
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    for i in range(61):
        if(result[0][i]==1):
            print (str(i))
            if i<10:
               a='0' + str(i)
            else:
                a = i
            img = cv2.imread('Training/000'+str(a)+'/1.ppm',0)
            cv2.imshow('Predicting',img)  
            k = cv2.waitKey(30)
            if k == 27:         # wait for ESC key to exit
                cv2.destroyAllWindows()
            #elif k == ord('s'): # wait for 's' key to save and exit
                #cv2.imwrite('messigray.png',img)
                #cv2.destroyAllWindows()                    


for i in range(20):
    a=randint(0, 61)
    if a<10:
        a='0' + str(a)
    try:    
        img = cv2.imread('Testing/000'+str(a)+'/1.ppm')
        test_image = image.load_img('Testing/000'+str(a)+'/1.ppm', target_size = (64, 64))
    except OSError as e:
        continue        
    test_image = image.img_to_array(test_image)
    predict(test_image)
    cv2.imshow('Testing',img)  
    k = cv2.waitKey(30)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    time.sleep(5)    
            
cv2.destroyAllWindows()
                         
vreader = imageio.get_reader('video.avi')
for i, img1 in enumerate(vreader):
    #gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    test_image = image.load_img(img1, target_size = (64, 64))
    test_image = cv2.resize(img1, (64, 64))    
    test_image = image.img_to_array(test_image)
    predict(test_image)
    cv2.imshow('video',img1)
    k=cv2.waitKey(30) & 0Xff
    if (k==27) :
        break

cap.release()
cv2.destroyAllWindows()
                      