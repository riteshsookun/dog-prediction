import pandas as pd
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import urllib.request

import numpy as np



dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))

IMG_SIZE = 224
image_01 = None

#MODEL_PATH = os.path.join(dir,'assets','model','2021_05_02_23_43_51_model_1.h5')
#model = load_model_web()

url = "https://westonemanor-westonemanorhote.netdna-ssl.com/assets/uploads/2021/05/2021_05_02_23_43_51_model_1.h5"
urllib.request.urlretrieve(
        url, 'model.h5')
MODEL_PATH = './model.h5'
model = load_model(MODEL_PATH)

LABELS_PATH = os.path.join(dir,'assets','labels','2021_05_02_23_43_51_labels.csv')

class Predictions:
    global image_01

    def load_convert_image(self, image_n):

        #img1 = Image.open(image_n)
        # opens image uploaded from GUI and resizes to expected size
        img1 = Image.open(image_n)

        img1 = img1.resize((IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img1)
        X = preprocess_input(x)

        return X

    def predictions_raw(self, pp_img):

        # ensure image is in the expected format
        # 1 = batch size, IMG_SIZE = 224, 3 = colour image

        predict = model.predict(np.array(pp_img).reshape((1, IMG_SIZE, IMG_SIZE, 3)))


        return predict

    def predictions_label(self):
        # pre-process
        predict_img = np.array(self.load_convert_image(image_01))

        # get labels
        labels = pd.read_csv(LABELS_PATH)
        labels = labels['breed']

        prediction = self.predictions_raw(predict_img)
        pred_index = np.argmax(prediction)
        pred_acc = np.amax(prediction)

        predicted_label = labels[pred_index]
        predicted_label = predicted_label.replace("_", " ").title()

        return predicted_label, pred_acc

if __name__ == "__main__":
    pass

