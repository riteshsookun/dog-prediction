from tensorflow.keras.models import load_model
import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


# update file locations here
model_name = "2021_05_12_17_15_55_model_2.h5"
pred_x_name = "2021_05_12_17_15_55_model_2_X_test_predictions.csv"
real_y_name = "2021_05_12_17_15_55_y_test.csv"
labels_name = "2021_05_12_17_15_55_labels.csv"
model_history_file_name = "2021_05_12_17_15_55_model_2_history.csv"

dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(dir,'assets')
model = load_model(os.path.join(DATASET_PATH,'model', model_name))
MODEL_TYPE = model_name.split("model",1)[1]
MODEL_TYPE = MODEL_TYPE[1]


def model_history(history):

    # plot loss
    plt.clf()
    plt.title('Cross Entropy Loss')
    plt.plot(history['loss'], color='blue', label='train')
    plt.plot(history['val_loss'], color='orange', label='val')
    plt.legend(['Train Loss', 'Val Loss'], loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #fig1.show()

    loss_file_name = "model_" + str(MODEL_TYPE) + "_loss_graph.png"
    plt.savefig(os.path.join(DATASET_PATH, 'classification_report', 'loss_acc_graphs', loss_file_name), dpi=300)

    # plot accuracy
    plt.clf()
    plt.title('Classification Accuracy')
    plt.plot(history['accuracy'], color='blue', label='train')
    plt.plot(history['val_accuracy'], color='orange', label='val')
    plt.legend(['Train Accuracy', 'Val Accuracy'], loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    #fig2.show()

    accuracy_file_name = "model_" + str(MODEL_TYPE) + "_accuracy_graph.png"
    plt.savefig(os.path.join(DATASET_PATH, 'classification_report', 'loss_acc_graphs', accuracy_file_name), dpi=300)

def prediction_label(prediction_array):
    global breed_names

    return breed_names[np.nanargmax(prediction_array)]

def confusion_matrix():
    global breed_names
    # get labels
    LABELS_PATH = os.path.join(DATASET_PATH,'labels', labels_name)
    labels_csv = LABELS_PATH
    labels = pd.read_csv(LABELS_PATH)
    breed_names = labels['breed']
    #
    # get X_test and y_test data
    X_test_path = os.path.join(DATASET_PATH,'test_predictions', pred_x_name)
    y_test_path = os.path.join(DATASET_PATH,'test_predictions', real_y_name)
    #
    X_predictions = np.genfromtxt(X_test_path, delimiter=",")
    y_predictions = np.genfromtxt(y_test_path, delimiter=",")



    #create pandas with the predicted values
    test_predictions = pd.concat([pd.DataFrame([prediction_label(X_predictions[i])], columns=['pred'])
                      for i in range(len(X_predictions))], ignore_index=True)

    # create pandas with the real values
    real_values = pd.concat([pd.DataFrame([breed_names[int(np.argwhere(y_predictions[i] == 1))]], columns=['real'])
                      for i in range(len(y_predictions))], ignore_index=True)

    test_predictions.reset_index(drop=True, inplace=True)
    real_values.reset_index(drop=True, inplace=True)

    # join two pandas together
    test_results = pd.concat([test_predictions, real_values], axis=1)

    print(test_results)

    cm_array_normalised = metrics.confusion_matrix(test_results['real'], test_results['pred'], normalize='true')

    confusion_matrix_file_name = "model_" + str(MODEL_TYPE) + "_confusion_matrix.png"
    df_cm_normalised = pd.DataFrame(cm_array_normalised, index = [i for i in breed_names],
                      columns = [i for i in breed_names])
    plt.figure(figsize=(30,25))
    sn.heatmap(df_cm_normalised, annot=True, cbar=False)
    plt.tight_layout()
    plt.savefig(os.path.join(DATASET_PATH, 'classification_report', confusion_matrix_file_name), dpi=800)


    classification_report = metrics.classification_report(test_results['real'], test_results['pred'], labels=breed_names,
                                                          output_dict=True)
    cf_df = pd.DataFrame(classification_report).transpose()
    classification_report_file_name = "model_" + str(MODEL_TYPE) + "_classification_report.csv"
    cf_df.to_csv(os.path.join(DATASET_PATH, 'classification_report', classification_report_file_name))


    print(len(test_results['real']))

def loss_acc_graphs():
    history = pd.read_csv(os.path.join(DATASET_PATH, 'model_history', model_history_file_name))
    model_history(history)


loss_acc_graphs()
confusion_matrix()
