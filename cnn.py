from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

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
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Trial/fifty.ppm', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices



if result[0][0] == 1:
    print ('dog')
else:
    print ('cat')
    

# To test Accuracy    
def test_accuracy(classifier, test_set, steps):
    num_correct = 0
    num_guesses = 0
    for i in range(steps):
        a = test_set.next()
        guesses = classifier.predict(a[0])
        correct = a[1]
        for index in range(len(guesses)):
            num_guesses += 1
            if round(guesses[index][0]) == correct[index]:
                num_correct += 1
    return num_correct, num_guesses 
    
test_accuracy(classifier,test_set,100)  
  