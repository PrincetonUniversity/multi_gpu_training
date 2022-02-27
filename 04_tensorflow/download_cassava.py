import tensorflow_datasets as tfds
_ = tfds.load(name='cassava', with_info=True, as_supervised=True, data_dir='.')
