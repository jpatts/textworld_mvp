import os, spacy
import tensorflow as tf
import numpy as np
from datetime import datetime

# Remove logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.enable_eager_execution()

# Hyperparams
vocab_size = 20200
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

# Load vocab
nlp = spacy.load('en')
with open('./vocab.txt') as f:
    vocab = f.read().split('\n')
# Create tokenized dicts
word2vec = {}
vec2word = {}
for i, w in enumerate(vocab):
    word2vec[w] = i
    vec2word[i] = w

x = np.array([word2vec['apple']])
one_hot = tf.one_hot([word2vec['pickup']], depth=vocab_size)

logits = model(x)
print(vec2word[tf.argmax(logits, axis=1).numpy()[0]])

# Create logger
extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = tf.contrib.summary.create_file_writer('/home/jojo/Documents/textworld_mvp/logs/' + extension)
writer.set_as_default()
global_step = tf.compat.v1.train.get_or_create_global_step()

# Train
for _ in range(epochs):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = tf.compat.v1.losses.softmax_cross_entropy(one_hot, logits)
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', loss)
        print(vec2word[tf.argmax(logits, axis=1).numpy()[0]])
    
    # Calculate and apply gradients
    grads = tape.gradient(loss, model.weights)
    optim.apply_gradients(zip(grads, model.weights))
    global_step.assign_add(1)
