import streamlit as st
import predictions


pred = predictions.Predictions()

# @st.cache
# def load_model_web():
#     url = "https://westonemanor-westonemanorhote.netdna-ssl.com/assets/uploads/2021/05/2021_05_02_23_43_51_model_1.h5"
#     model_file = file_io.FileIO(url, mode='rb')
#     return model_file

def main():

    st.title("What breed is this dog?")

    # Add a selectbox to the sidebar:
    add_selectbox = st.sidebar.selectbox(
        'How would you like to be contacted?',
        ('Email', 'Home phone', 'Mobile phone')
    )

    # Add a slider to the sidebar:
    add_slider = st.sidebar.slider(
        'Select a range of values',
        0.0, 100.0, (25.0, 75.0)
    )

    # get image to classify
    predictions.image_01 = st.file_uploader("Upload photo", type=["png", "jpg", "jpeg"])


    # only show image and prediction once photo has been uploaded
    if predictions.image_01:
        st.write('Image')
        # show image
        st.image(predictions.image_01)

        # get classification label
        try:
            prediction_label, prediction_acc = pred.predictions_label()

            if prediction_acc < 0.50:

             # display classification output
                st.write("We're not too sure what breed your dog is but "
                         "our best guess is a ", prediction_label, " with a probability of",
                         ('%.0f' % (prediction_acc * 100.0)), "%")
            else:
                st.write("Your dog is most likely a ", prediction_label, " with a probability of",
                         ('%.0f' % (prediction_acc * 100.0)), "%")

        except ValueError:
            st.error("We couldn't process your image! Please try another photo")

if __name__ == "__main__":
    main()
