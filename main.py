import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("my_model70.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Soil Recognition"])

#Main Page
if(app_mode=="Home"):
    st.header("SOIL CLASSIFICATION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Soil Recognition System! 🌿🔍
    
    Our mission is to help in identifying Soil type efficiently. Upload an image of the soil, and our system will analyze it to detect the soil type.

    ### How It Works
    1. **Upload Image:** Go to the **Soil Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential Soil category.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate results.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Soil Recognition** page in the sidebar to upload an image and experience the power of our Soil Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 8K rgb images of various types of soil which is categorized into 8 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    
                #### Content
                1. train (5098 images)
                2. test (2979 images)

                """)

#Prediction Page
elif(app_mode=="Soil Recognition"):
    st.header("Soil Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Alluvial Soil', 'Black Soil', 'Chalky Soil', 'Clay Soil', 'Mary Soil', 'Red Soil', 'Sand', 'Silt']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))