from Decoder import *
from Encoder import *
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

f = open('training_dataset/train.ast.src', 'r')
input_data = str(f.readline())
# print(input_data)
tokenizer_inputs = Tokenizer(num_words=len(input_data), filters='')
tokenizer_inputs.fit_on_texts(input_data)
f.close()


input_sequences = tokenizer_inputs.texts_to_sequences(input_data)

input_max_len = max(len(s) for s in input_sequences)
print('Max Input Length: ', input_max_len)

# print(input_data)
print(input_sequences)

word2idx_inputs = tokenizer_inputs.word_index
print('Found %s unique input tokens.' % len(word2idx_inputs))

tokenizer_outputs = Tokenizer(num_words=len(input_data), filters='')
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))


num_words_output = len(word2idx_outputs) + 1
num_words_inputs = len(word2idx_inputs) + 1

idx2word_inputs = {v:k for k, v in word2idx_inputs.items()}
idx2word_outputs = {v:k for k, v in word2idx_outputs.items()}


f = open('training_dataset/train.txt.tgt', 'r')
target_data = str(f.readline())
# print(input_data)
tokenizer_target = Tokenizer(num_words=len(input_data), filters='')
tokenizer_target.fit_on_texts(target_data)
f.close()


target_sequences_input = tokenizer_inputs.texts_to_sequences(target_data)

target_max_len = max(len(s) for s in input_sequences)
print('Max Input Length: ', input_max_len)



encoder_inputs = pad_sequences(input_sequences, maxlen=input_max_len, padding='post')
print("encoder_inputs.shape:", encoder_inputs.shape)
print("encoder_inputs[0]:", encoder_inputs[0])

decoder_inputs = pad_sequences(target_sequences_input, maxlen=target_max_len, padding='post')
print("decoder_inputs[0]:", decoder_inputs[0])
print("decoder_inputs.shape:", decoder_inputs.shape)

# decoder_targets = pad_sequences(target_sequences, maxlen=target_max_len, padding='post')
