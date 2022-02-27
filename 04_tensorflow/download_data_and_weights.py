import tensorflow as tf
import tensorflow_datasets as tfds

# download the data (4 GB) on the login node
_ = tfds.load(name='cassava', with_info=True, as_supervised=True, data_dir='.')

# download the model weights on the login node
_ = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
