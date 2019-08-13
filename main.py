import os
import tensorflow as tf
import numpy as np
from datetime import datetime

# Remove logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.enable_eager_execution()

# Hyperparams
vocab_size = 2
embedding_size = 1
epochs = 100

# Create model
class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size):
        super(Model, self).__init__()

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.out = tf.keras.layers.Dense(vocab_size)
    
    def call(self, x):
        return self.out(self.embedding(x))

model = Model(vocab_size, embedding_size)
optim = tf.keras.optimizers.Adam(0.01)

# Create dataset
input_output_mapping = {'apple': 'pickup'}
word2vec = {0:'apple', 1:'pickup'}
x = np.array([0])
one_hot = np.array([[0, 1]])

logits = model(x)
print(word2vec[tf.argmax(logits, axis=1).numpy()[0]])

# Create logger
extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = tf.contrib.summary.create_file_writer('/home/jojo/Documents/textworld_mvp/logs/' + extension)
writer.set_as_default()
global_step = tf.compat.v1.train.get_or_create_global_step()

# Train
for _ in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = tf.compat.v1.losses.softmax_cross_entropy(one_hot, logits, reduction='none')
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', loss)
        print(word2vec[tf.argmax(logits, axis=1).numpy()[0]])
    
    # Calculate and apply gradients
    grads = tape.gradient(loss, model.weights)
    optim.apply_gradients(zip(grads, model.weights))
    global_step.assign_add(1)

