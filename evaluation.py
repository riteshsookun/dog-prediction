import tensorflow
from tensorflow.keras.models import load_model
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image





dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(dir,'assets')
model_name = "2021_05_10_17_33_09_model_3.h5"
model = load_model(os.path.join(DATASET_PATH,'model', model_name))
MODEL_TYPE = model_name.split("model",1)[1]
MODEL_TYPE = MODEL_TYPE[1]


def model_history(history):
    # plot loss

    plt.title('Cross Entropy Loss')
    plt.plot(history['loss'], color='blue', label='train')
    plt.plot(history['val_loss'], color='orange', label='val')
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # plot accuracy
    plt.title('Classification Accuracy')
    plt.plot(history['accuracy'], color='blue', label='train')
    plt.plot(history['val_accuracy'], color='orange', label='val')
    plt.legend(['Train Accuracy', 'Val Accuracy'], loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


def prediction_label(prediction_array):
    return breed_names[np.nanargmax(prediction_array)]


# get labels
LABELS_PATH = os.path.join(DATASET_PATH,'labels','2021_05_10_17_33_09_labels.csv')
labels_csv = LABELS_PATH
labels = pd.read_csv(LABELS_PATH)
breed_names = labels['breed']
#
# get X_test and y_test data
X_test_path = os.path.join(DATASET_PATH,'test_predictions', '2021_05_10_17_33_09_model_3_X_test_predictions.csv')
y_test_path = os.path.join(DATASET_PATH,'test_predictions', '2021_05_10_17_33_09_y_test.csv')
#
X_predictions = np.genfromtxt(X_test_path, delimiter=",")
y_predictions = np.genfromtxt(y_test_path, delimiter=",")


# print(X_predictions[0])
# print(y_predictions[0])
# pred_index = np.argmax(X_predictions[0])
# print(pred_index)
# #
# y_index = int(np.argwhere(y_predictions[0] == 1))
# actual = breed_names[y_index]
#
# print(pred_index)
# print(y_index)
# print(actual)
# #######





test_predictions = pd.concat([pd.DataFrame([prediction_label(X_predictions[i])], columns=['pred'])
                  for i in range(len(X_predictions))], ignore_index=True)

real_values = pd.concat([pd.DataFrame([breed_names[int(np.argwhere(y_predictions[i] == 1))]], columns=['real'])
                  for i in range(len(y_predictions))], ignore_index=True)

test_predictions.reset_index(drop=True, inplace=True)
real_values.reset_index(drop=True, inplace=True)

test_results = pd.concat([test_predictions, real_values], axis=1)

print(test_results)

cm_array = metrics.confusion_matrix(test_results['real'], test_results['pred'])

print(metrics.confusion_matrix)
confusion_matrix_file_name = "model_" + str(MODEL_TYPE) + "_confusion_matrix.png"
df_cm = pd.DataFrame(cm_array, index = [i for i in breed_names],
                  columns = [i for i in breed_names])
plt.figure(figsize=(30,25))
sn.heatmap(df_cm, annot=True, cbar=False)
plt.tight_layout()
plt.savefig(os.path.join(DATASET_PATH, 'classification_report', confusion_matrix_file_name), dpi=800)

classification_report = metrics.classification_report(test_results['real'], test_results['pred'], labels=breed_names,
                                                      output_dict=True)
cf_df = pd.DataFrame(classification_report).transpose()
classification_report_file_name = "model_" + str(MODEL_TYPE) + "_classification_report.csv"
cf_df.to_csv(os.path.join(DATASET_PATH, 'classification_report', classification_report_file_name))


print(len(test_results['real']))
print(len(test_results['pred']))
""""""


#


model_history_file_name = "2021_05_10_17_33_09_model_3_history.csv"
history = pd.read_csv(os.path.join(DATASET_PATH, 'model_history', model_history_file_name))
model_history(history)
