# TensorFlow

The starting point for [multi-GPU training with Keras](https://www.tensorflow.org/tutorials/distribute/keras) is `tf.distribute.MirroredStrategy`. In this approach, the model is copied to `N` GPUs and gradients are synced as we saw previously. Be sure to use [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) to handle data loading as is done in the example on this page and is explained graphically [here](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/guide/data_performance.ipynb#scrollTo=i3NtGI3r-jLp).

## Single-Node, Synchronous, Multi-GPU Training

Here were train the ResNet-50 model on the Cassava dataset (see [video](https://www.youtube.com/watch?v=xzSCvXDcX68) on TensorFlow YouTube channel).

### Step 1: Installation

Install TensorFlow for the V100 nodes on Adroit (see [these directions](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow#install) for all other cases including the A100 node on Adroit):

```bash
# procedure for adroit v100 nodes only
$ ssh <YourNetID>@adroit.princeton.edu
$ module load anaconda3/2021.11
$ conda create --name tf2-v100 tensorflow-gpu tensorflow-datasets --channel conda-forge -y
```

### Step 2: Download the Data

This example using the `cassava` dataset which requires 4 GB of storage space. You should do this on `/scratch/network/<YourNetID>` or `/scratch/gpfs/<YourNetID>` and not in `/home.`

Run the commands below to download the data (4 GB in size):

```
$ cd multi_gpu_training/04_tensorflow
$ conda activate tf2-v100
(tf2-v100) $ python download_data_and_weights.py
```

### Step 3: Inspect the Script

Below is the contents of `mnist_classify.py`:

```python
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
```

### Step 4: Submit the Job

Below is a sample Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=cassava       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3/2021.11
conda activate tf2-v100

python cassava_classify.py --batch-size-per-replica=32 --epochs=15
```

Note that `srun` is not called and there is only one task. Submit the job as follows:

```
(tf2-v100) $ sbatch job.slurm
```

### Performance

The training time is shown below for different choices of `cpus-per-task` and the number of GPUs:

| nodes         | ntasks        | cpus-per-task | GPUs    | Training Time (s) |
|:-------------:|:-------------:|:------------:|:--------:|:-----------------:|
| 1             |     1         | 4            |  1       | xxx               |
| 1             |     1         | 8            |  1       | xxx               |
| 1             |     1         | 16           |  1       | xxx               |
| 1             |     1         | 4            |  2       | 367 (339)         |
| 1             |     1         | 8            |  2       | 344 (334)         |
| 1             |     1         | 16           |  2       | 343 (332)         |
| 1             |     1         | 4            |  3       | 268 (256)         |
| 1             |     1         | 8            |  3       | 263 (251)         |
| 1             |     1         | 16           |  3       | 261 (249)         |
| 1             |     1         | 8            |  4       | 233 (220)         |
| 1             |     1         | 16           |  4       | 228 (214)         |
| 1             |     1         | 32           |  4       | 233 (218)         |

All runs were done on adroit-h11g1 while making certain that no other jobs were running on the node:

```
#SBATCH --mem=770000M
#SBATCH --nodelist=adroit-h11g1
```

## Multi-node Training

Look to [MultiWorkerMirroredStrategy](https://www.tensorflow.org/guide/distributed_training#multiworkermirroredstrategy) for using the GPUs on more than one compute node. There is an example for the [Keras API](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras). Consider using Horovod instead of this approach (see below).

## Horovod

[Horovod](https://horovod.ai/) is a distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet. It is based on MPI.
