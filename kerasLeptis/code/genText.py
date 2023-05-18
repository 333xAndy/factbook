import tensorflow as tf
import tensorflow.python.keras as keras
import numpy as np
import os 
import time
import module

def text_from_id(ids):
    return tf.strings.reduce_join(chars_from_ids(ids),axis=-1)

def split_input_target(sequencce):
    input_text = sequencce[:-1]
    target_text=sequencce[1:]
    return input_text, target_text

#load the data
fullPath = os.path.abspath("./" + 'grand_Mosque.txt')
path_to_file = tf.keras.utils.get_file('grand_Mosque.txt', 'file://'+fullPath)

#read the data
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#print(f'Length of text: {len(text)} characters')

#unique characters in file
vocab = sorted(set(text))
#print(f'{len(vocab)} unique characters')

#process text, note to self, change text to vocab if not right
chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
id_Chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
ids = id_Chars(chars)

chars_from_ids = tf.keras.layers.StringLookup(vocabulary=id_Chars.get_vocabulary(), invert=True, mask_token=None)
chars=chars_from_ids(ids)

#join back into strings
arr = text_from_id(ids)

"""
Predicting text. RNN maintains a single direction state that depends on previous seen elements
Divide into sequence, each sequence has a target sequence
"""
all_ids = id_Chars(tf.strings.unicode_split(text, 'UTF-8'))
id_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length=100
sequences = id_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

#Before training data, we need to shuffle and pack into batches
BATCH_SIZE = 64
BUFFER_SIZE = 1000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

dataset

#output layer with size outputs, likelihood of each character
vocab_size = len(id_Chars.get_vocabulary())
#input layer with lookup table to map each char id to vector with dim dimensions
embedding_dim=128
#type of rnn with size
rnn_units = 512

model = module.MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)

#how far predicted values deviate from the actual values
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

#implements adam algorithm, gradient descent method that is based on adaptive estimates of
#first order or second order moments
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#number of iterations for training data in use.
EPOCHS = 750

#callback to save the keras model or model weights at some frequency
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

#model.load_weights(checkpoint_dir)

one_step_model = module.OneStep(model, chars_from_ids, id_Chars)

start = time.time()
states = None
next_char = tf.constant(['Grand Mosque:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
