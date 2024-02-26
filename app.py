import streamlit as st
import numpy as np
import base64
import util

# Load the saved artifacts
util.load_saved_artifacts()


# Define the Streamlit app
def main():
    st.title("Sports Celebrity Image Classification")
    # Display placeholders for three images
    st.header("Upload Images for Classification")

    # Define image paths and captions
    image_paths = ["images/hardik pandya.jpg", "images/rohit sharma.jpg", "images/virat kholi.jpg"]
    captions = ["HARDIK PANDIYA", "ROHIT SHARMA", "VIRAT KOHLI"]

    # Display images in one line with equal size and captions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image_paths[0], caption=captions[0], use_column_width=True)
    with col2:
        st.image(image_paths[1], caption=captions[1], use_column_width=True)
    with col3:
        st.image(image_paths[2], caption=captions[2], use_column_width=True)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        # Display the uploaded image
        image = uploaded_file.read()
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img_str = base64.b64encode(image)
        # Predict button
        if st.button('Predict'):
            # Perform classification
            result = util.classify_image(img_str)
            # Display classification result
            st.write("Prediction Results:")
            if result is None:
                st.write("Error occurred during classification.")
            else:
                for res in result:
                    st.write(f"Class: {res['class']}")
                    st.write(f"Probability: {res['class_probability']}")


# Function to convert image to base64 string
def get_base64_string(img):
    buffered = np.array(img)
    img_str = base64.b64encode(buffered).decode("utf-8")
    return img_str


if __name__ == '__main__':
    main()
