import numpy as np
import os
import cv2

from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import Flatten, Dense, Dropout, LocallyConnected2D, Input, Multiply, Lambda, MaxPooling2D
from keras.optimizers import Adam

# File parameters
RED_FOLDER = './images/red/'
YELLOW_FOLDER = './images/yellow/'
GREEN_FOLDER = './images/green/'

# RED_MASKS = './masks/red/'
# YELLOW_MASKS = './masks/yellow/'
# GREEN_MASKS = './masks/green/'

# Training parameters
lr = 0.01
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08
decay = 0.0

epochs = 3
batch_size = 16
WINDOWS_PER_IMG = 1036

def GetFiles():
    files_red = os.listdir(RED_FOLDER)
    # masks_red = os.listdir(RED_MASKS)
    for i in range(len(files_red)):
        files_red[i] = RED_FOLDER + files_red[i]
        # masks_red[i] = RED_MASKS + masks_red[i]

    files_yellow = os.listdir(YELLOW_FOLDER)
    # masks_yellow = os.listdir(YELLOW_MASKS)
    for i in range(len(files_yellow)):
        files_yellow[i] = YELLOW_FOLDER + files_yellow[i]
        # masks_yellow[i] = YELLOW_MASKS + masks_yellow[i]

    files_green = os.listdir(GREEN_FOLDER)
    # masks_green = os.listdir(GREEN_MASKS)
    for i in range(len(files_green)):
        files_green[i] = GREEN_FOLDER + files_green[i]
        # masks_green[i] = GREEN_MASKS + masks_green[i]

    index_red = len(files_red)
    index_yellow = index_red + len(files_yellow)
    index_green = index_yellow + len(files_green)

    # Target = [isRed, isGreen]
    target = np.zeros( (index_green, 2) )
    target[:index_red, 0] = 1
    target[index_yellow : index_green, 1] = 1

    # Single column target values (used @ StratifiedShuffleSplit)
    strats = np.ones((index_green, 1))
    strats[:index_red] = 0 * strats[:index_red]
    strats[index_yellow : index_green] = 2 * strats[index_yellow : index_green]

    # Concatenate files
    files = np.array(files_red + files_yellow + files_green)
    # masks = np.array(masks_red + masks_yellow + masks_green)

    # Shuffle items
    sss = StratifiedShuffleSplit(test_size=0.07)
    for train_index, test_index in sss.split(files, strats):
        X_train, X_test = files[train_index], files[test_index]
        y_train, y_test = target[train_index,:], target[test_index,:]
        # m_train, m_test = masks[train_index], masks[test_index]

    # return X_train, y_train, m_train, X_test, y_test, m_test
    return X_train, y_train, X_test, y_test

def Detector():
    model = Sequential()
    model.add( Conv2D(32, (5, 5), input_shape=(60,80,3), strides = (5,5), padding='same', activation='relu') )
    model.add( Conv2D(64, (3, 3), strides = (2,2), padding='same', activation='relu') )
    model.add( Conv2D(128, (3, 3), strides = (2,2), padding='same', activation='relu') )
    model.add( Conv2D(256, (3, 3), strides = (1,1), padding='valid', activation='relu') )
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, use_bias = True, activation='hard_sigmoid'))

    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    return model

def Detector2():
    nn_input = Input(shape=(600,800,3)) 
    l1 = Conv2D(16, (20, 20), input_shape=(600,800,3), strides = (10,10), padding='same', activation='relu')(nn_input)
    l2 = Conv2D(32, (10, 10), strides = (5,5), padding='same', activation='relu')(l1)
    l3= Conv2D(128, (8, 8), strides = (4,4), padding='valid', activation='relu')(l2)

    l4 = LocallyConnected2D(128, (1, 1), strides = (1,1), padding='valid', activation='relu')(l3)

    l5 = Conv2DTranspose(128, (8,8), strides = (4,4), padding='valid', activation='relu')(l4)
    l5_connect = LocallyConnected2D(128, (1, 1), strides = (1,1), padding='valid', activation='relu')(l2)
    mul5 = Multiply()([l5, l5_connect])
    
    l6 = Conv2DTranspose(32, (10,10), strides = (5,5), padding='same', activation='relu')(mul5)
    # l6_connect = Conv2D(32, (1, 1), strides = (1,1), padding='valid', activation='relu')(l1)
    l6_connect = LocallyConnected2D(32, (1, 1), strides = (1,1), padding='valid', activation='relu')(l1)
    mul6 = Multiply()([l6, l6_connect])
    
    output = Conv2DTranspose(1, (20,20), strides = (10,10), padding='same', activation='hard_sigmoid', use_bias = True)(mul6) 
    

    model = Model(inputs=nn_input, outputs=output)
    print(model.summary())

    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    #model.compile(loss='binary_crossentropy', optimizer=adam)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def Detector3():
    model = Sequential()
    model.add( Conv2D(32, (20, 20), input_shape=(600,800,3), strides = (10,10), padding='same', activation='relu') )
    model.add( Conv2D(64, (10, 10), strides = (5,5), padding='same', activation='relu') )
    model.add( Conv2D(128, (8, 8), strides = (4,4), padding='valid', activation='relu') )
    model.add( LocallyConnected2D(256, (1, 1), strides = (1,1), padding='valid', activation='relu') )
    model.add( Conv2DTranspose(128, (8, 8), strides = (4,4), padding='valid', activation='relu') )
    model.add( Conv2DTranspose(64, (10, 10), strides = (5,5), padding='same', activation='relu') )
    model.add( Conv2DTranspose(1, (20,20), strides = (10,10), padding='same', activation='hard_sigmoid', use_bias = True) )
    print(model.summary())

    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def Classifier():
    model = Sequential()
    model.add( Conv2D(32, (5, 5), input_shape=(600,800,3), strides = (3,3), padding='same', activation='relu') )
    model.add( Conv2D(64, (5, 5), strides = (3,3), padding='same', activation='relu') )
    model.add( Conv2D(128, (5, 5), strides = (3,3), padding='same', activation='relu') )
    model.add( Conv2D(256, (3, 3), strides = (3,3), padding='same', activation='relu') )
    model.add( Conv2D(512, (3, 3), strides = (3,3), padding='same', activation='relu') )
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, use_bias = True, activation='softmax'))

    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    return model

def Classifier2():
    model = Sequential()
    model.add( Conv2D(32, (3, 3), input_shape=(400,400,3), strides = (1,1), padding='same', activation='relu') )
    model.add( Conv2D(32, (3, 3), strides = (1,1), padding='same', activation='relu') )
    model.add( MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add( Conv2D(32, (3, 3), strides = (1,1), padding='same', activation='relu') )
    model.add( MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add( Conv2D(32, (3, 3), strides = (1,1), padding='same', activation='relu') )   
    model.add( MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add( Conv2D(32, (3, 3), strides = (1,1), padding='same', activation='relu') )
    model.add( MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add( Conv2D(64, (3, 3), strides = (1,1), padding='same', activation='relu') )
    model.add( MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add( Conv2D(64, (3, 3), strides = (1,1), padding='same', activation='relu') )
    model.add( MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add( Conv2D(64, (3, 3), strides = (1,1), padding='same', activation='relu') )
    model.add( MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Dense(2, use_bias = True, activation='softmax'))

    adam = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())

    return model

def GenDetect(X, y, batch):
    features = np.zeros( (batch, 600, 800, 3) )
    labels = np.zeros( (batch, 600, 800, 1) )
    
    index = 0
    #count = 0
    total = int(len(X) / batch)
    while True:
        for i in range(batch):
            index = (index + 1) % total
            features[i] = LoadImage(X[index])
            labels[i,:,:,0] = LoadMask(y[index])
        yield features, labels

def GenClassify(X, y, batch):
    features = np.zeros( (batch, 400, 400, 3) )
    labels = np.zeros( (batch, 2) )
    
    index = 0
    total = int(len(X) / batch)
    while True:
        for i in range(batch):
            index = (index + 1) % total
            features[i] = LoadImage(X[index])
            labels[i] = y[index]
        yield features, labels

def GenWindow(X,y):
    size = X.size
    index = 0

    while True:
        img = LoadImage(X[index])
        mask = LoadMask(y[index])

        features = np.zeros( (WINDOWS_PER_IMG, 60, 80, 3) )
        labels = np.zeros( (WINDOWS_PER_IMG, 1))

        count = 0
        for i in range(30,571,10):
            for j in range(40, 761, 20):
                features[count] = img[i-30:i+30, j-40:j+40, :]
                window = mask[i-30:i+30, j-40:j+40]
                
                avg = np.average(window)
                if avg > 0.33:
                    labels[count] = 1
                else:
                    labels[count] = 0
        index = (index + 1) % size
        yield features, np.uint8(labels)
    

def LoadImage(filename):
    try:
        img = cv2.imread(filename)
        img = cv2.resize(img[:,100:-100], (400,400))
        return np.array(img.astype('float32') / 255.0)
    except:
        print(filename)

def LoadMask(filename):
    img = LoadImage(filename)
    return np.average(img, axis=-1)

def Binarize(X):
    X[X < 0.5] = 0
    X[X > 0.5] = 1
    return X

def Main():
    X_train, y_train, m_train, X_test, y_test, m_test = GetFiles()
    model = Classifier2() 

    steps = int(X_train.size / batch_size)
    model.fit_generator(GenClassify(X_train, y_train, batch_size), steps_per_epoch = steps, epochs = epochs, verbose = 1)

    steps = int(X_test.size / batch_size)
    loss = model.evaluate_generator(GenClassify(X_test, y_test, batch_size), steps)
    print("Test metric: {}".format(loss))
    pred = model.predict_generator(GenClassify(X_test, y_test, batch_size), steps)
    pred[pred < 0.5] = 0
    pred[pred >= 0.5] = 1

    diff = abs(pred - y_test[:32,:])
    print("SCORE: ", diff.sum(), " | 32")

    with open("model.json", "w") as json_file:
        json_file.write(model.to_json())
    
    model.save_weights("model.h5")

    """
    return

    model = Detector2()

    steps = int(X_train.size / batch_size)
    model.fit_generator(GenDetect(X_train, m_train, batch_size), steps_per_epoch = steps, epochs = epochs, verbose = 1, max_queue_size=1)

    X = np.zeros((1, 600, 800, 3))
    X[0] = LoadImage(X_test[0])
    mask = model.predict(X)

    img = cv2.imread(X_test[0])
    cv2.imwrite('./test.jpg', img)

    img[:,:,0] = np.multiply(img[:,:,0],mask[0,:,:,0]) 
    img[:,:,1] = np.multiply(img[:,:,1],mask[0,:,:,0]) 
    img[:,:,2] = np.multiply(img[:,:,2],mask[0,:,:,0]) 

    cv2.imwrite('./test_masked.jpg', img)

    img[:,:,0] = np.uint8(255*mask[0,:,:,0]) 
    img[:,:,1] = np.uint8(255*mask[0,:,:,0]) 
    img[:,:,2] = np.uint8(255*mask[0,:,:,0])  
    cv2.imwrite('./test_mask.jpg', img)

    with open("model_detect.json", "w") as json_file:
        json_file.write(model.to_json())
    
    model.save_weights("model_detect.h5")

    return
    steps = X_train.size
    model.fit_generator(GenWindow(X_train, m_train), steps_per_epoch = steps, epochs = epochs)

    preds = model.predict_generator(GenWindow(X_test, m_test), steps=1)
    img = LoadImage(X_test[0])
    output = np.zeros_like(img)

    count = 0
    for i in range(30,571,10):
        for j in range(40, 761, 20):
            if preds[count] > 0.5:
                output[i-30:i+30, j-40:j+40, :] = img[i-30:i+30, j-40:j+40, :]

    cv2.imwrite('test.jpg', output)

    with open("model_windows.json", "w") as json_file:
        json_file.write(model.to_json())
    
    model.save_weights("model_windows.h5")
    return

    print("Compiling Detection Model")
    model = Detector2()
    print("Done.")

    steps = int(X_train.size / batch_size)
    model.fit_generator(GenDetect(X_train, m_train, batch_size), steps_per_epoch = steps, epochs = epochs, verbose = 1, max_queue_size=1)

    X = np.zeros((1, 600, 800, 3))
    X[0] = LoadImage(X_test[0])
    mask = model.predict(X)

    img = cv2.imread(X_test[0])
    cv2.imwrite('./test.jpg', img)

    img[:,:,0] = np.multiply(img[:,:,0],mask[0,:,:,0]) 
    img[:,:,1] = np.multiply(img[:,:,1],mask[0,:,:,0]) 
    img[:,:,2] = np.multiply(img[:,:,2],mask[0,:,:,0]) 

    cv2.imwrite('./test_masked.jpg', img)

    img[:,:,0] = np.uint8(255*mask[0,:,:,0]) 
    img[:,:,1] = np.uint8(255*mask[0,:,:,0]) 
    img[:,:,2] = np.uint8(255*mask[0,:,:,0])  
    cv2.imwrite('./test_mask.jpg', img)

    with open("model_detect.json", "w") as json_file:
        json_file.write(model.to_json())
    
    model.save_weights("model_detect.h5")
    """


if __name__ == '__main__':
    Main()