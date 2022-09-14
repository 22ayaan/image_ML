
from operator import sub
import streamlit as st
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import shutil
import time
import os
import subprocess
import zipfile

from datetime import datetime
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def save_uploadedfile(uploadedfile):
     with open(os.path.join("uploaded_dataset/",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to uploaded_dataset".format(uploadedfile.name))

def imageCheck(model,img):
    img = tf.keras.utils.load_img(
        img, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    st.success(
        "**This image most likely belongs to {} with a {:.2f} % confidence.**"
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    

def uploadOption(upload_method):
    if upload_method == "Upload Image from device":
        img = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        upload_button = st.button("Upload")
        if upload_button:
            img = Image.open(img)
                
            if img.format == 'PNG':
                img = img.convert('RGB')
                img.save('image.jpg')
            st.image(img,width=250)

            date = datetime.now().strftime("%Y_%m_%d-%I-%M-%S_%p")
            directory = 'uploaded_images/'
            filepath = directory+'image_' + date + '.jpg'
            img.save(filepath)

    elif upload_method == "Enter the URL of image":
        img_url = st.text_input("Enter the URL of the image you want to test: ")
        #upload_button = st.button("Upload")
        # if upload_button:
        st.image(img_url,width=250)
        st.success("Image successfully uploaded!")
        img_path = tf.keras.utils.get_file(origin=img_url)
        filepath = pathlib.Path(img_path)
    return filepath

st.header("Image Classification Model Dashboard")

# tab1, tab2, tab3 = st.tabs(['Flower image classification model', 'Train your own model', 'Test your model'])

# with tab1:
upload_method = st.radio("Select upload option", ("Upload Image from device", "Enter the URL of image"))

batch_size = 32
img_height = 180
img_width = 180  
class_names = []

loaded_model = tf.keras.models.load_model('models/model#2')
data_dir = pathlib.Path('flower_photos')
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    
class_names = val_ds.class_names
try:
    file_path = uploadOption(upload_method)
    imageCheck(loaded_model,file_path)
except:
    st.info("Waiting for image to be uploaded")

# with tab2:
#     st.header("Train your own model")
#     upload_method2 = st.radio("Select upload option", ("Upload training Dataset ZIP file from device", "Enter the URL of dataset source"))
#     dataset_path = ""
#     if upload_method2 == "Upload training Dataset ZIP file from device":
#         zip_file = st.file_uploader("Upload a dataset ZIP file", type=["zip"])
#         if zip_file is not None:
#             save_uploadedfile(zip_file)
#         upload_button_zip = st.button("Upload", key="upload_dataset")
#         if upload_button_zip:
#             subprocess.check_output(["tar","-xf", "uploaded_dataset/"+zip_file.name])
#             # fullpath = os.path.abspath(os.path.join("uploaded_dataset/"))
#             # st.write("Uploaded file: "+fullpath)
#             # dataset_dir = tf.keras.utils.get_file(
#             #     fname=zip_file.name,
#             #     origin='file:\\' + fullpath,
#             #     untar=True,
#             #     extract=False,
#             #     archive_format='auto'
#             # )
#             dataset_path = pathlib.Path(zip_file.name.split('.')[0])
#             st.write(dataset_path)
#             st.success("Dataset uploaded successfully")

#     elif upload_method2 == "Enter the URL of dataset source":
#         dataset_url = st.text_input("Enter the URL of the dataset source: ")
#         upload_button_url = st.button("Upload", key="upload_url")

#         if upload_button_url:
#             st.success("Dataset uploaded successfully")
#             dataset_path = tf.keras.utils.get_file(origin=dataset_url)
#             dataset_path = pathlib.Path(dataset_path)

    
#     st.write("pressed")        
#     # if dataset_path is not None:
#     val_split, train_split = st.select_slider(
#         "Validation split", 
#         options=[20,30,40,50,60,70,80], 
#         value= (20,80), 
#         help='Recommended: 20-80 or 30-70')
#     st.write("before compile")
#     st.write(dataset_path)
#     if st.button("Compile Model"):
#         train_ds = tf.keras.utils.image_dataset_from_directory(
#             dataset_path,
#             validation_split=val_split/100,
#             subset="training",
#             seed=123,
#             image_size=(img_height, img_width),
#             batch_size=batch_size)

#         val_ds_tab2 = tf.keras.utils.image_dataset_from_directory(
#             dataset_path,
#             validation_split=val_split/100,
#             subset="validation",
#             seed=123,
#             image_size=(img_height, img_width),
#             batch_size=batch_size)
                
#         class_names = train_ds.class_names

#         AUTOTUNE = tf.data.AUTOTUNE

#         train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
#         val_ds_tab2 = val_ds_tab2.cache().prefetch(buffer_size=AUTOTUNE)

#         normalization_layer = layers.Rescaling(1./255)

#         normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
#         image_batch, labels_batch = next(iter(normalized_ds))

#         num_classes = len(class_names)

#         data_augmentation = keras.Sequential(
#             [
#             layers.RandomFlip("horizontal",
#                                 input_shape=(img_height,
#                                             img_width,
#                                             3)),
#             layers.RandomRotation(0.1),
#             layers.RandomZoom(0.1),
#             ]
#         )

#         trained_model = Sequential([
#             data_augmentation,
#             layers.Rescaling(1./255),
#             layers.Conv2D(16, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(),
#             layers.Conv2D(32, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(),
#             layers.Conv2D(64, 3, padding='same', activation='relu'),
#             layers.MaxPooling2D(),
#             layers.Dropout(0.2),
#             layers.Flatten(),
#             layers.Dense(128, activation='relu'),
#             layers.Dense(num_classes)
#         ])

#         trained_model.compile(optimizer='adam',
#                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                         metrics=['accuracy'])

#         st.success("Model compiled successfully")
#     # except NameError:
#     #     st.info("Waiting for dataset to be uploaded")
#         summary_button = st.button("View Summary")
#         if summary_button:
#             st.write("MODEL SUMMARY:\n", trained_model.summary())

#         epochs = st.number_input("Number of epochs",help="Recommended: >=15")
#         if st.button("Train Model", key="train_model"):
#             with st.spinner("Training model..."):
#                 st.write("Enjoy some music while your model cooks up...")
#                 st.audio("sakura-hz-watching-anime.mp3")
#                 try:
#                     history = trained_model.fit(
#                     train_ds,
#                     validation_data=val_ds,
#                     epochs=epochs)
#                 except NameError:
#                     st.info("")
#             st.success("Model trained successfully")
#             st.balloons()

#             model_name = st.text_input("Enter the name of the model: " )
#             if st.button("Save Model"):
#                 model_path = 'models/' + model_name + datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '.h5'
#                 trained_model.save(model_path)
#                 st.success("Model saved successfully")
#             try:
#                 st.download_button(label="Download model", data=trained_model ,file_name=model_name + '.h5')
#                 st.success("Model downloaded successfully")
#             except NameError:
#                 st.info("")

# with tab3:
#     st.header("Test your model")
#     upload_method4 = st.radio("Select upload option", ("Upload model from device", "Use saved model"))

#     if upload_method4 == "Upload model from device":
#         model_file = st.file_uploader("Upload a model file", type=["hdf5"])

#         upload_button2 = st.button("Upload", key="upload_model")
#         if upload_button2:
#             model = tf.keras.models.load_model(model_file)
#             st.success("Model uploaded successfully")
#     elif upload_method4 == "Use saved model":
#         model_name = st.text_input("Enter the name of the model: ")
#         model_path = 'models/' + model_name
#         model = tf.keras.models.load_model(model_path)
#         st.success("Model loaded successfully")

#     st.subheader("Let's test the model...")

#     upload_method3 = st.radio(
#         "Select upload option", 
#         ("Upload Image from device", 
#         "Enter the URL of image", 
#         "Use device camera to click an image"))

#     upload_button3 = st.button("Upload", key="upload_image")

#     if upload_button3:
#         if upload_method3 == "Upload Image from device" or upload_method3 == "Enter the URL of image":
#             file_path = uploadOption(upload_method3)
#             imageCheck(loaded_model,file_path)
#         else:
#             camera_image = st.camera_input("Click an image to test the model")
#             if camera_image:
#                 st.image(camera_image)
#                 imageCheck(model, camera_image)
#     # st.write("Report a wrong classification")
#     # report_button = st.button("Report")

#     # if report_button:
#     #     with st.form:
#     #         st.subheader("Report a wrong classification")
#     #         st.markdown("Please enter the details of the wrong classification and help us improve your model")
#     #         input_img = Image.open(file_path)
#     #         st.image(input_img, width=250)
#     #         correct_class = st.text_input("What is the correct classification of the object in the image?")
