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

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection

#classifier.add(Dense(units = 128, activation = 'relu'))
#classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.add(Dense(units=62, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Training/',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('Testing/',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 142.96,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 78.75)
                         
classifier.save('model.h5')  

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
            img = cv2.imread('Training/000'+str(i)+'/1.ppm',0)
            cv2.imshow('img',img)  
            k = cv2.waitKey(0)
            if k == 27:         # wait for ESC key to exit
                cv2.destroyAllWindows()
            #elif k == ord('s'): # wait for 's' key to save and exit
                #cv2.imwrite('messigray.png',img)
                #cv2.destroyAllWindows()                    

                         
vreader = imageio.get_reader('video.avi')
for i, img1 in enumerate(vreader):
    #gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #test_image = image.load_img(img1, target_size = (64, 64))
    test_image = cv2.resize(img1, (64, 64))    
    test_image = image.img_to_array(test_image)
    predict(test_image)
    cv2.imshow('video',img1)
    k=cv2.waitKey(30) & 0Xff
    if (k==27) :
        break

cap.release()
cv2.destroyAllWindows()
                         
                         
        
# Part 3 - Making new predictions
test_image = image.load_img('image1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
for i in range(61):
    if(result[0][i]==1):
        print (str(i))
training_set.class_indices.size
    

