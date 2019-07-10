"""
Copyright 2019 Nadheesh Jihan

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import warnings

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.layers import GRU
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import precision_recall_fscore_support
from termcolor import colored

warnings.filterwarnings("ignore")

nltk.download('stopwords')
# one hot encode

seed = 64
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.DataFrame()
df = pd.read_csv('../../../data/triple_with_stv.csv')
lines = df['triple'].values.tolist()
stv = df['stv'].values.reshape(-1, 1)
truth = df['truth'].values

triple_lines = [line.split() for line in lines]

EMBEDDING_DIM = 200

# Vectorize the text samples into a S2 integer tensor
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(triple_lines)
sequences = tokenizer_obj.texts_to_sequences(triple_lines)

# pad sequences : add padding to make all the vectors of same length

# define vocabulary size
vocab_size = len(tokenizer_obj.word_index) + 1

print(colored(sequences, 'green'))

# pad sequences
word_index = tokenizer_obj.word_index
max_length = 5  # should be 3 for Yago

triple_pad = np.array(sequences)

num_words = len(word_index) + 1

print(colored(num_words, 'cyan'))

input1 = layers.Input(shape=(max_length,))
embedding = layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length)(input1)
pooling = GRU(units=64, dropout=0.2, recurrent_dropout=0.2)(embedding)

input2 = layers.Input(shape=(1,))
concat = layers.Concatenate(axis=-1)([pooling, input2])

# l1 = layers.Dense(64, activation='relu')(concat)
# dropout = layers.Dropout(0.1)(l1)
out = layers.Dense(1, activation='sigmoid')(concat)

model = models.Model(inputs=[input1, input2], outputs=[out])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Split the data into training set and validation set
VALIDATION_SPLIT = 0.3

indices = np.arange(triple_pad.shape[0])
np.random.shuffle(indices)
triple_pad = triple_pad[indices]
truth = truth[indices]
num_validation_samples = int(VALIDATION_SPLIT * triple_pad.shape[0])

X_train_pad = triple_pad[:-num_validation_samples]
X_train_psl = stv[:-num_validation_samples]
y_train = truth[:-num_validation_samples]

X_test_pad = triple_pad[-num_validation_samples:]
X_test_psl = stv[-num_validation_samples:]
y_test = truth[-num_validation_samples:]

print('Shape of X_train_pad tensor: ', X_train_pad.shape)
print('Shape of y_train tensor: ', y_train.shape)
print('Shape of X_test_pad tensor: ', X_test_pad.shape)
print('Shape of y_test tensor: ', y_test.shape)

print(colored('Training...', 'green'))

history = model.fit([X_train_pad, X_train_psl], y_train, batch_size=128, epochs=12,
                    validation_data=([X_test_pad, X_test_psl], y_test), verbose=2)

y_pred = (model.predict(x=[X_test_pad, X_test_psl]) > 0.3).astype(np.int32)
metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print()
print(colored("Precision: ", 'green'), colored(metrics[0], 'blue'))
print(colored("Recall: ", 'green'), colored(metrics[1], 'blue'))
print(colored("F1: ", 'green'), colored(metrics[2], 'blue'))

import matplotlib.pyplot as plt

plt.style.use('ggplot')


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


plt.show(plot_history(history))
