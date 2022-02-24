# TensorFlow

The starting point for [multi-GPU training with Keras](https://www.tensorflow.org/tutorials/distribute/keras) is `tf.distribute.MirroredStrategy`. In this approach, the model is copied to `N` GPUs and gradients are synced. Be sure to use `[tf.data](https://www.tensorflow.org/api_docs/python/tf/data)` to handle data loading.

## Single-Node, Synchronous, Multi-GPU Training with `tf.distribute.MirroredStrategy`

Install TensorFlow for the V100 nodes:

```
$ ssh <YourNetID>@adroit.princeton.edu
$ module load anaconda3/2021.11
$ conda create --name tf2-gpu tensorflow-gpu tensorflow-datasets -y
```

### Step 2: Download the Data

```
$ python
Python 3.9.7 (default, Sep 16 2021, 13:09:58) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow_datasets as tfds
>>> datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
```

### Step 3: Submit the Job

Below is a sample Slurm:

```bash
#!/bin/bash
#SBATCH --job-name=tf2-test      # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4G per cpu-core is default)
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2021.11
conda activate tf2-gpu

python mnist_classify.py
```

Note that `srun` is not called and there is only one task. Submit the job as follows:

```
$ sbatch job.slurm
```

Below are some performance numbers:



[See here]()

## Multi-node Training

## Horovod

