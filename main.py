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
units = 1024
dropout = 0.0

class Seq2seq(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, units, dropout):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(vocab_size, embedding_size, units, dropout)
        self.decoder = Decoder(vocab_size, embedding_size, units, dropout)
        self.vocab_size = vocab_size
    
    def call(self, enc_in, max_cmd_len, teacher, batch_size):
        self.encoder.reset_hidden(batch_size)
        # Get encoded data and hidden state
        encoded, enc_hidden = self.encoder(enc_in)
        
        # Init hidden state
        dec_hidden = enc_hidden
        # Init output data structures
        logits = tf.zeros((batch_size, 1, self.vocab_size))
        predictions = tf.zeros((batch_size, 1), dtype=tf.dtypes.int64)
        # For timestep in max timesteps
        for t in range(max_cmd_len):
            # Get input for timestep
            dec_in = tf.expand_dims(teacher[:, t], 1)
            # Get logits at timestep
            logits_t, dec_hidden, attn = self.decoder(dec_in, dec_hidden, encoded)
            # Save logits
            logits = tf.concat([logits[:, :t], tf.expand_dims(logits_t, 1)], 1)
            # Save prediction
            pred = tf.reshape(tf.argmax(input=logits_t, axis=1), [batch_size, 1])
            predictions = tf.concat([predictions[:, :t], pred], 1)
        
        return logits, predictions


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, enc_units, dropout):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru = tf.keras.layers.GRU(enc_units, dropout=dropout, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def call(self, x_in):
        # x_in shape = (batch_size, num_entites * max_entity_length)
        embedded = self.embedding(x_in)
        encoded, self.enc_hidden = self.gru(embedded, initial_state=self.enc_hidden)
        # (batch_size, num_entities * max_entity_length, units), (batch_size, units)
        return encoded, self.enc_hidden
    
    def reset_hidden(self, batch_size):
        self.enc_hidden = tf.zeros((batch_size, self.enc_units))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size, dec_units, dropout):
        super(Decoder, self).__init__()
        self.attention = BahdanauAttention(dec_units)
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        self.gru = tf.keras.layers.GRU(dec_units, dropout=dropout, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, x_in, enc_hidden, encoded):
        # (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(enc_hidden, encoded)

        # (batch_size, 1, embedding_dim)
        embedded = self.embedding(x_in)

        # (batch_size, 1, embedding_dim + hidden_size)
        concatenated = tf.concat([tf.expand_dims(context_vector, 1), embedded], axis=-1)

        decoded, dec_hidden = self.gru(concatenated)
        
        # (batch_size * 1, hidden_size)
        decoded = tf.reshape(decoded, (-1, decoded.shape[2]))

        # (batch_size, vocab)
        x_out = self.fc(decoded)

        return self.dropout(x_out), dec_hidden, attention_weights


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, enc_hidden, encoded):
    # (batch_size, max_length, 1)
    score = self.V(tf.nn.tanh(
        self.W1(encoded) + self.W2(tf.expand_dims(enc_hidden, 1))))

    # (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # (batch_size, hidden_size)
    context_vector = tf.reduce_sum(input_tensor=attention_weights * encoded, axis=1)

    return context_vector, attention_weights
    

model = Seq2seq(vocab_size, embedding_size, units, dropout)
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

x = np.array([[word2vec['apple']]])
y = np.array([[word2vec['pickup']]])
one_hot = tf.one_hot([[word2vec['pickup']]], depth=vocab_size)

logits, pred = model(x, 1, y, 1)
print(vec2word[pred[0][0].numpy()])

# Create logger
extension = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = tf.contrib.summary.create_file_writer('/home/jojo/Documents/textworld_mvp/logs/' + extension)
writer.set_as_default()
global_step = tf.compat.v1.train.get_or_create_global_step()

# Train
for _ in range(epochs):
    with tf.GradientTape() as tape:
        logits, pred = model(x, 1, y, 1)
        loss = tf.compat.v1.losses.softmax_cross_entropy(one_hot, logits)
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('loss', loss)
        print(vec2word[pred[0][0].numpy()])
    
    # Calculate and apply gradients
    grads = tape.gradient(loss, model.weights)
    optim.apply_gradients(zip(grads, model.weights))
    global_step.assign_add(1)
