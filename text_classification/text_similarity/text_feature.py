import tensorflow as tf
import tensorflow_hub as hub
import os

TENSORBOARD_FOLDER = os.getcwd()+'/__tensorboard__'

tf.logging.set_verbosity(tf.logging.INFO)

embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")
embeddings = embed(["cat is on the mat", "dog is in the fog"])

print(embeddings)