import numpy as np
import tensorflow as tf
import tensorboard as tb
import pandas as pd
import os
from tensorboard.plugins import projector


# Set up a logs directory, so Tensorboard knows where to look for files.
log_dir='logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# Load the data in trainLabels.csv
data = pd.read_csv('trainLabels.csv')

# load the features in completeFeaturesTrainDataset.csv
features = pd.read_csv('completeFeaturesTrainDataset.csv')

# drop the first column (index)
features = features.drop(features.columns[0], axis=1)
# drop the first column (id)
features = features.drop(features.columns[0], axis=1)
# drop the last column (class)
features = features.drop(features.columns[-1], axis=1)

# save the last column (size)
#last_column = features[features.columns[-1]]

# normalize the features with the size in the last column
features = features.apply(lambda x: x/x[-1], axis=1)
# drop the last column (size)
features = features.drop(features.columns[-1], axis=1)


# saving to a csv file
features.to_csv('features.csv', index=False)


with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
    f.write('Id\tClass\n')
    for i in range(len(data)):
        f.write(f'{data["Id"][i]}\t{data["Class"][i]}\n')
    f.close()

# Create a checkpoint from embedding, the filename and key are the
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=tf.Variable(features))
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.sprite.image_path = '/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/logs/sprite_image.png'
embedding.sprite.single_image_dim.extend([64, 64, 3])
embedding.metadata_path = 'metadata.tsv'
projector.visualize_embeddings(log_dir, config)

# saving a checkpoint
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

# to launch this tensorborad use the command:
# tensorboard --logdir=logs
