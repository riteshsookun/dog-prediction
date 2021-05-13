
# this file isn't part of the GUI! This file is used to create, train and save the ML models that are used later.
import os

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

import pandas as pd
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from contextlib import redirect_stdout
import datetime


model = ''
current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

""" ML Parameters"""
IMG_SIZE = 224  # IMG_SIZE = 224
BATCH_SIZE = 10  # 20
EPOCHS = 65  # 65
LR = 0.00001
MODEL_TYPE = 0
""""""""""""""""""""


# plot diagnostic learning curves
def model_history(history):
    # plot loss
    plt.title('Cross Entropy Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='orange', label='test')
    plt.legend(['Test Loss', 'Val Loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    history_file_name_loss_png = current_time + "_model_" + str(MODEL_TYPE) + "_loss_plot.png"
    #plt.savefig(os.path.join(model_history_path, history_file_name_loss_png), dpi=800)
    #plt.show()

    # plot accuracy
    plt.title('Classification Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='orange', label='test')
    history_file_name_accuracy_png = current_time + "_model_" + str(MODEL_TYPE) + "_accuracy_plot.png"
    #plt.savefig(os.path.join(model_history_path, history_file_name_accuracy_png), dpi=800)
    #plt.show()

    # save val accuracy to file
    acc_val_df = pd.DataFrame(history.history)

    history_file_name_csv = current_time + "_model_" + str(MODEL_TYPE) + "_history.csv"
    acc_val_df.to_csv(os.path.join(model_history_path, history_file_name_csv))

def load_convert_image(list_of_images, preprocess_type):
    X = []

    for image_n in list_of_images:
        img1 = (image.load_img(image_n, target_size=(IMG_SIZE, IMG_SIZE)))
        x = image.img_to_array(img1)

        if preprocess_type == 1:
            # preprocess the same way used for ResNet50
            x = tensorflow.keras.applications.resnet50.preprocess_input(x)
        if preprocess_type == 2:
            # preprocess the same way used for VGG16
            x = tensorflow.keras.applications.vgg16.preprocess_input(x)

        if preprocess_type == 3:
            # preprocess the same way used for InceptionV3
            x = tensorflow.keras.applications.inception_v3.preprocess_input(x)

        if preprocess_type == 4:
            # preprocess for the baseline CNN model
            x = x/255.0

        X.append(x)

    return X


def preprocess_datasets():
    global X_train, X_val, X_train_pp, X_val_pp, model
    X_train_pp = np.array(load_convert_image(X_train, MODEL_TYPE))
    X_val_pp = np.array(load_convert_image(X_val, MODEL_TYPE))

    X_train_pp = X_train_pp.astype('float32')
    X_val_pp = X_val_pp.astype('float32')

    if MODEL_TYPE == 1:
        # Creates resnet50 model
        model = resnet50()
    if MODEL_TYPE == 2:
        # creates VGG16 model
        model = build_model_VGG()

    if MODEL_TYPE == 3:
        model = inceptionResNetV3()

    if MODEL_TYPE == 4:
        model = model_scratch()


def preprocess_test_dataset():
    global X_test, X_test_pp
    X_test_pp = np.array(load_convert_image(X_test, MODEL_TYPE))
    X_test_pp = X_test_pp.astype('float32')


def model_scratch():
    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', use_bias=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(BatchNormalization(axis=3, scale=False))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization(axis=3, scale=False))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization(axis=3, scale=False))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
    model.add(BatchNormalization(axis=3, scale=False))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(120, activation='softmax'))

    # compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


    return model


def build_model_VGG():
    print("VGG16")
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(2048, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dense(1024, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dense(512, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dropout(0.3))

    # output layer
    model.add(Dense(120, activation='softmax'))

    # freeze all layers in VGG16
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model
    model.compile(optimizers.Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def resnet50():
    print("ResNet50")
    base_model = ResNet50(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(2048, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dense(1024, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dense(512, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dropout(0.3))

    # output layer
    model.add(Dense(120, activation='softmax'))

    #freeze all layers in ResNet50
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model
    model.compile(optimizers.Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def inceptionResNetV3():
    print("InceptionResNetV3")
    base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(2048, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dense(1024, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dense(512, activation='relu', use_bias=False, kernel_initializer='uniform'))
    model.add(Dropout(0.3))

    # output layer
    model.add(Dense(120, activation='softmax'))

    # freeze all layers in Inception
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model
    model.compile(optimizers.Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


print("TF version:", tf.__version__)

# Check for GPU availability
print("GPU", "available" if tf.config.list_physical_devices("GPU") else "not")

dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(dir,'assets','dataset')
ASSETS_PATH = os.path.join(dir,'assets')

model_history_path = os.path.join(ASSETS_PATH,'model_history')

# import labels for the photos
LABELS_PATH = os.path.join(DATASET_PATH, 'file_list.csv')
labels_csv = pd.read_csv(LABELS_PATH, encoding='utf8')
labels_csv.columns = ['id', 'breed']

# display a few labels
print(labels_csv.head())

# How many images are there of each breed?
breed_names = labels_csv['breed'].value_counts().index.tolist()

# prints the first 10 breeds
print(breed_names[:10])

# Saves the labels that will be needed when predicting
labels_csv_file_name = current_time + '_labels.csv'
LABELS_PATH = os.path.join(ASSETS_PATH,'labels', labels_csv_file_name)
pd.DataFrame(breed_names, columns=['breed']).to_csv(LABELS_PATH, index=False)

# Show a graph of the distribution of breeds in the dataset
plt.bar(breed_names, labels_csv['breed'].value_counts().tolist())
plt.xticks(rotation=90)
plt.xlabel("Class Name")
plt.ylabel("Number of images")
plt.title("Number of images per class")
plt.tight_layout()
plt.show()

# creates the file path for the location of the dataset images
dataset = pd.DataFrame(columns=['filename'])
for i, label in zip(range(0,len(labels_csv)+1), labels_csv["id"]):
 dataset.loc[i,'filename'] = os.path.join(DATASET_PATH, 'Images', label)

# this contains the location of the images
dataset = dataset['filename']


# checking if the number of images in the dataset matches the actual amount of files
if len(os.listdir(DATASET_PATH + r"\Images")) == len(dataset):
    print("No discrepancies found between the actual amount of files and amount in dataset ")
else:
    print(
        "One or more discrepancies found. Check dataset directory"
        "Number of images in directory: ", len(os.listdir(DATASET_PATH + r"\Images")), ". \nnumber of"
                                                                                   "images in dataset: ", len(dataset)
    )

# checking if number of labels match the number of unique filenames
if len(breed_names) == len(dataset):
    print("No discrepancies found between the labels and the number of unique filenames")
else:
    print(
        "One or more discrepancies found between the labels and number of unique filenames.",
        "Number of labels: ", len(breed_names), "and number of unique filenames (i.e. breeds): ", len(dataset)
    )

# show random image (image will be from index 100-5000)
# img = mpimg.imread(dataset[random.randint(100,5000)])
# imgplot = plt.imshow(img)
# ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# plt.show()

# ensures breed_names is in numpy array
breed_names = np.array(breed_names)

# Turn every label into a boolean array
labels = labels_csv.breed.values
one_hot_encode = [label == breed_names for label in labels]

print(one_hot_encode[1])

# printing image
print(dataset[1])

# classifying the predictors and target variables as X and Y
# X contains the image dataset
# Y contains the labels which tells the ML model which class the images from X belong to
X = dataset
y = one_hot_encode

# Split dataset into training and test 70:30
NUM_IMAGES = len(dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                  random_state=42, shuffle= True)
# splits training dataset to create validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                  random_state=42, shuffle= True)

# variables to hold converted data for later
X_train_pp = 0
X_val_pp = 0
X_test_pp = 0

ntrain = len(X_train)
nval = len(X_val)
print("X_train: ", ntrain, "X_val: ", nval, "y_train: ", len(y_train), "y_val: ", len(y_val),
      "X_test: ", len(X_test), "y_test: ", len(y_test))

# making sure y_train and y_val are numpy arrays
y_train = np.array(y_train)
y_val = np.array(y_val)

# # Let's have a look on our training data
# plt.figure(figsize=(20,10))
# column = 5
# for i in range(column):
#     plt.subplot(5/column + 1, column, i + 1)
#     plt.imshow(X_train[i])
#     print(y_train[i])
#
# plt.show()

# check shape of training data
print("batch size, height, width and channels")
print("\nshape of x train images is: ", X_train.shape)
print("shape of x val images is: ", X_val.shape)
print("shape of y train labels is: ", y_train.shape)
print("shape of y val labels is: ", y_val.shape)


for k in range(1,5):

    # sets MODEL_TYPE to value from 1-4
    MODEL_TYPE = k
    # pre-process datasets
    preprocess_datasets()

    # summarise the model
    model.summary()
    # export model summary
    MODEL_SUMMARY_PATH = os.path.join(ASSETS_PATH, 'model_summary')
    model_summary_file_name = current_time + "_model_" + str(MODEL_TYPE) + "_model_summary.txt"
    with open(os.path.join(MODEL_SUMMARY_PATH, model_summary_file_name), 'w') as f:
        with redirect_stdout(f):
            model.summary()

    train_datagen = ImageDataGenerator(
                                       horizontal_flip=True,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2 ,
                                       rotation_range=30,
                                       shear_range=0.1,
                                       fill_mode='constant'
                                        )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train_pp, y_train, batch_size=BATCH_SIZE)
    val_generator = val_datagen.flow(X_val_pp, y_val, batch_size=BATCH_SIZE)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    MODEL_SAVE_PATH = os.path.join(ASSETS_PATH,'model')

    model_file_name = current_time + "_model_" + str(MODEL_TYPE) + ".h5"
    mcp = ModelCheckpoint(os.path.join(MODEL_SAVE_PATH,
                                       model_file_name), monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)

    history = model.fit_generator(train_generator,
                                  steps_per_epoch= ntrain//BATCH_SIZE,
                                  epochs= EPOCHS,
                                  validation_data= val_generator,
                                  validation_steps= nval//BATCH_SIZE,
                                  callbacks=[es, mcp],
                                  verbose=1)

    #run model_history function to save data about model
    model_history(history)

    "TEST DATASET"
    # pre-process test dataset
    # 1 is resnet, 2 is VGG, 3 is inception
    # calculating results using test set
    # remember that our test sets are X = X_test and y = y_test
    preprocess_test_dataset()

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(X_test_pp, y_test, batch_size=BATCH_SIZE)

    t_predictions = model.predict(X_test_pp, verbose=1)

    # Saves the X_test data needed for evaluation later
    TEST_PRED_PATH = os.path.join(ASSETS_PATH,'test_predictions')
    t_predictions_file_name = current_time + "_model_" + str(MODEL_TYPE) + "_X_test_predictions.csv"
    np.savetxt(os.path.join(TEST_PRED_PATH, t_predictions_file_name), t_predictions, delimiter=",")



# Saves the y_test data needed for evaluation later
TEST_PRED_PATH = os.path.join(ASSETS_PATH, 'test_predictions')
y_test_file_name = current_time + "_y_test.csv"
np.savetxt(os.path.join(TEST_PRED_PATH, y_test_file_name), y_test, delimiter=",")
