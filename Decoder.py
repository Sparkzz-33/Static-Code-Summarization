import keras
import tensorflow as tf
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
  
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
  
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, input_sequence, state):

        embed = self.embedding(input_sequence)

        lstm_out, state_h, state_c = self.lstm(embed, state)

        logits = self.dense(lstm_out)

        return logits, state_h, state_c