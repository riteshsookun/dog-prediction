import streamlit as st
import predictions


pred = predictions.Predictions()


def main():

    # sidebar help
    st.sidebar.title('Need some help?')
    st.sidebar.write("To get started, make sure you already have the"
                     " photo of the dog you want to classify saved to your computer.")
    st.sidebar.write("Next, click on 'Browse files then select the location of the saved image. The image"
                     " must either be a .PNG, .JPG or .JPEG file and no larger than 200MB.")
    st.sidebar.write("Once you upload your photo, the predicted breed should show up on screen! The"
                     " probability score is also displayed.")
    st.sidebar.write("We can only know about 120 different breeds so we might not be able to correctly guess all"
                     " types of breeds.")
    st.sidebar.write("If for some reason we cannot process your image, an error message"
                     " will come up asking for another photo :)")
    # title
    st.title("What breed is this dog?")

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
            print(prediction_acc)
            if prediction_acc < 0.50:
             # display classification output
                st.write("We're not too sure what breed your dog is but "
                         "our best guess is a ", prediction_label, " with a probability of",
                         ('%.0f' % (prediction_acc * 100.0)), "%")
            elif prediction_acc > 0.99:
                # rounding to 2dp here as we don't want to say 100% accuracy since it's just a prediction
                st.write("Your dog is most likely a ", prediction_label, " with a probability of",
                         ('%.2f' % (prediction_acc * 100.0)), "%")
            else:
                st.write("Your dog is most likely a ", prediction_label, " with a probability of",
                         ('%.0f' % (prediction_acc * 100.0)), "%")

        except ValueError:
            # catch all ValueErrors
            st.error("We couldn't process your image! Please try another photo")


if __name__ == "__main__":
    main()