import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

def imageCheck(model):
    image_url = input("Enter the URL of the image you want to test: ")
    image_path = tf.keras.utils.get_file(fname = None, origin=image_url)
    img = tf.keras.utils.load_img(
        image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

batch_size = 32
img_height = 180
img_width = 180  
class_names = []

loaded_model = tf.keras.models.load_model('model#2')
data_dir = pathlib.Path('flower_photos')
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
    
class_names = val_ds.class_names

imageCheck(loaded_model)