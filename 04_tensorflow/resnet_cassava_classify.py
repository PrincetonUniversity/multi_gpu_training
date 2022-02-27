import argparse
import os
import tensorflow_datasets as tfds
import tensorflow as tf
from time import perf_counter

def preprocess_data(image, label):
  image = tf.image.resize(image, (300, 300))
  image = tf.cast(image, tf.float32) / 255.0
  return image, label

def create_dataset(batch_size_per_replica, datasets, strategy):
  batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
  return datasets['train'].map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE) \
                          .cache() \
                          .shuffle(1000) \
                          .batch(batch_size) \
                          .prefetch(tf.data.AUTOTUNE)

def create_model(num_classes):
  base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
  x = base_model.output
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  x = tf.keras.layers.Dense(1016, activation="relu")(x)
  predictions = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
  model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
  return model

def train(epochs, num_classes, train_dataset, strategy):
  with strategy.scope():
    model = create_model(num_classes)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  metrics=['accuracy'])

    start_time = perf_counter()
    model.fit(train_dataset, epochs=epochs)
    print("Training time:", perf_counter() - start_time)
  return None

def print_info(num_replicas_in_sync, batch_size_per_replica, info, num_classes):
  print(f'TF Version: {tf.__version__}')
  print(f'Number of GPUs: {num_replicas_in_sync}')
  print(f'Batch size per GPU: {batch_size_per_replica}')
  print(f'Train records: {info.splits["train"].num_examples}')
  print(f'Test records:  {info.splits["test"].num_examples}')
  print(f'Number of classes: {num_classes}')
  return None

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Multi-GPU Training Example')
  parser.add_argument('--batch-size-per-replica', type=int, default=32, metavar='N',
                      help='input batch size for training (default: 32)')
  parser.add_argument('--epochs', type=int, default=15, metavar='N',
                      help='number of epochs to train (default: 15)')
  args = parser.parse_args()
  
  datasets, info = tfds.load(name='cassava', with_info=True, as_supervised=True, data_dir=".")
  num_classes = info.features["label"].num_classes

  strategy = tf.distribute.MirroredStrategy()
  print_info(strategy.num_replicas_in_sync, args.batch_size_per_replica, info, num_classes)
  train_dataset = create_dataset(args.batch_size_per_replica, datasets, strategy)
  train(args.epochs, num_classes, train_dataset, strategy)
