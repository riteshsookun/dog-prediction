import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import numpy as np
import streamlit as st

IMG_SIZE = 224
image_01 = None

dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(dir,'assets')


@st.cache(allow_output_mutation=True)
def load_saved_model():
    model_name = "2021_05_10_17_33_09_model_3.h5"
    model = load_model(os.path.join(DATASET_PATH,'model', model_name))
    return model

LABELS_PATH = (os.path.join(dir,'assets','labels','2021_05_10_17_33_09_labels.csv'))
model = load_saved_model()

class Predictions:
    global image_01

    def load_convert_image(self, image_n):

        # opens image uploaded from GUI and resizes to expected size
        img1 = Image.open(image_n)

        img1 = img1.resize((IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img1)
        X = tensorflow.keras.applications.inception_v3.preprocess_input(x)

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
