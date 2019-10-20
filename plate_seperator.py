import cv2
from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np
from matplotlib import pyplot as plt 


from keras import layers
from keras import models
from keras import optimizers
from keras.utils import plot_model
from keras import backend

path = os.path.dirname(os.path.realpath(__file__)) + "/"
NUMBER_OF_LABELS = 36

#Where image is an OpenCV array
def convertToPIL(image):
    return Image.fromarray(image)

#Takes in an image and returns a list of PIL images
#image is the path to the image that is in question
def splitImage(image):
    img = cv2.imread(image)
    height = img.shape[0]
    width = img.shape[1]

    imWidth = 102

    topY = 80
    bottomY = 260

    firstX = 46
    secondX = 148
    thirdX = 350
    fourthX = 452


    # cv2.rectangle(img, (firstX, topY), (firstX+imWidth, bottomY), (255,0,0), 1)
    # cv2.rectangle(img, (secondX, topY), (secondX+imWidth, bottomY), (255,0,0), 2)
    # cv2.rectangle(img, (thirdX, topY), (thirdX+imWidth, bottomY), (255,0,0), 2)
    # cv2.rectangle(img, (fourthX, topY), (fourthX+imWidth, bottomY), (255,0,0), 2)

    splitted = []

    letterOne = img[topY:bottomY, firstX:firstX+imWidth]
    letterTwo = img[topY:bottomY, secondX:secondX+imWidth]
    numberOne = img[topY:bottomY, thirdX:thirdX+imWidth]
    numberTwo = img[topY:bottomY, fourthX:fourthX+imWidth]

    splitted.append(convertToPIL(letterOne))
    splitted.append(convertToPIL(letterTwo))
    splitted.append(convertToPIL(numberOne))
    splitted.append(convertToPIL(numberTwo))


    # cv2.imshow('Test image',numberTwo)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return splitted

def prepareData():
    files = os.listdir(path+"pictures/")
    #List that contains an array that holds the input and expected output as a number
    datalist = []

    for file in files:
        letters = splitImage(path + "pictures/" + file)

        #Get the plate number as a string
        plateNumber = file.replace(".png","").replace("plate_","")
        
        for i in range (0, 4):
            #Assign output values (0-9 = 0-9, A-Z = 10-36)
            numberLetter = ord(plateNumber[i]) - 48
            if (numberLetter > 9):
                numberLetter = numberLetter - 7

            singledata = np.array([np.array(letters[i]), numberLetter])
            datalist.append(singledata)
    
    alldata = np.array(datalist)
    np.random.shuffle(alldata)

    #Split the data
    X_dataset_orig = np.array([data[0] for data in alldata[:]])
    Y_dataset_orig = np.array([data[1] for data in alldata]).T

    #We normalize all of the pixels in each image to a value between 0 and 1
    X_dataset = X_dataset_orig / 255
    #Change the Y dataset to a one hot
    Y_dataset = np.eye(NUMBER_OF_LABELS)[Y_dataset_orig]

    return X_dataset, Y_dataset

def reset_weights(model):
    session = backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def train(X_dataset, Y_dataset):
    conv_model = models.Sequential()

    conv_model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 102, 3)))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    conv_model.add(layers.MaxPooling2D((2, 2)))
    conv_model.add(layers.Flatten())
    conv_model.add(layers.Dropout(0.5))
    conv_model.add(layers.Dense(512, activation='relu'))
    conv_model.add(layers.Dense(36, activation='softmax'))

    conv_model.summary()

    LEARNING_RATE = 1e-4
    conv_model.compile(loss='categorical_crossentropy',optimizer=optimizers.RMSprop(lr=LEARNING_RATE),metrics=['acc'])

    reset_weights(conv_model)

    history_conv = conv_model.fit(X_dataset, Y_dataset, validation_split=0.2, epochs=10, batch_size=16)

    plt.plot(history_conv.history['loss'])
    plt.plot(history_conv.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss'], loc='upper left')
    plt.show()

    plt.plot(history_conv.history['acc'])
    plt.plot(history_conv.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy (%)')
    plt.xlabel('epoch')
    plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
    plt.show()

    conv_model.save('plate_seperator.h5')

def list_to_string(list):
    plate = ""
    for c in list:
        if c < 10:
            plate = plate + str(c)
        else:
            plate = plate + str(unichr(c+55))
    
    return plate

def test_nn():
    files = os.listdir(path+"pictures/")
    model = models.load_model('plate_seperator.h5')

    total_correct=0
    i = 0
    for file in files:
        i=i+1
        print(file)
        letters_pil = splitImage(path + "pictures/" + file)
        plate_number = file.replace(".png","").replace("plate_","")
        letters = np.array([np.array(letter) for letter in letters_pil]) / 255
        prediction = model.predict_classes(letters)

        plate = list_to_string(prediction)
        print("Predicted: " + plate + " // Actual: " +plate_number)
        
        if plate == plate_number:
            total_correct = total_correct + 1

    print("Percentage Correct: "+ str(float(total_correct)/i * 100)+"%")



#Main
# X_dataset, Y_dataset = prepareData()
# train(X_dataset, Y_dataset)
test_nn()

            
