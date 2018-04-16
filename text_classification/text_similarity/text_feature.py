import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

TENSORBOARD_FOLDER = os.getcwd()+'/__tensorboard__'
HUB_MODULE_FOLDER = os.getcwd()+'/hub_module/'

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    logits = tf.layers.dense(net, 2, activation=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'net': net,
            'output': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def eval_input_fn(features, labels=None, batch_size=1):
    """An input function for evaluation or prediction"""   
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

tf.logging.set_verbosity(tf.logging.INFO)

# Load text feature extractor
if os.path.isdir(HUB_MODULE_FOLDER+'nnlm-en-dim128'):
  embedded_text_feature_column = hub.text_embedding_column(
    key="sentence", 
    module_spec=HUB_MODULE_FOLDER+'nnlm-en-dim128')
else:
  embedded_text_feature_column = hub.text_embedding_column(
    key="sentence", 
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

# Build estimator for prediction
classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': embedded_text_feature_column
    }
)

# fake train
classifier.train(input_fn=lambda:({"sentence": [""]},[0]), steps=1)

predictions = classifier.predict(
    input_fn=lambda:eval_input_fn({"sentence": ["cat is on the mat", "cat is on the mat"]})
    )

output = []
for pred_dict, expect in zip(predictions, [0, 0]):
    template = ('\nPrediction is "{}", \n{}')

    net = pred_dict['net']
    output.append(pred_dict['net'])

output = np.array(output)

# put data to T_SNE
from tensorflow.contrib.tensorboard.plugins import projector

if not os.path.isdir(TENSORBOARD_FOLDER):
    os.mkdir(TENSORBOARD_FOLDER)

metadata = os.path.join(TENSORBOARD_FOLDER, 'metadata.tsv')

images = tf.Variable(output, name='output')

with open(metadata, 'w') as metadata_file:
    for row in range(2):
        c = row
        metadata_file.write('{}\n'.format(c))

with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(TENSORBOARD_FOLDER, 'images.ckpt'))

    
    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(TENSORBOARD_FOLDER, sess.graph), config)
 


    #print(template.format(net, output))

'''
classifier.train(
    input_fn=lambda:{"sentence": ["cat is on the mat", "dog is in the fog"]},
    steps=1)

tables_initializer = tf.tables_initializer()

with tf.Session() as sess:
	sess.run(tables_initializer)
	sess.run(embeddings)
'''

'''
net = tf.feature_column.input_layer({"sentence": ["cat is on the mat", "dog is in the fog"]}, embedded_text_feature_column)

tables_initializer = tf.tables_initializer()
with tf.Session() as sess:
    sess.run(tables_initializer)
    sess.run(net)
'''	

