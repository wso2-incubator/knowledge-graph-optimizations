import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pandas import read_csv
from matplotlib import pyplot
import numpy as np
import gensim
import pandas as pd
import os
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, KeyedVectors
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from termcolor import colored
from keras.utils import to_categorical
import tensorflow as tf
from keras import layers

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
# one hot encode

# load dataset
dataset = read_csv('../../../data/yago_triple_with_stv.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0,1]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
# pyplot.show()

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


df = pd.DataFrame()
df = pd.read_csv('../../../data/yago_triple_with_stv.csv')
sentence_lines = list()
lines = df['triple'].values.tolist()
stv = df['stv'].values.tolist()

for line in lines:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    sentence_lines.append(words)

print('Number of lines', len(sentence_lines))
EMBEDDING_DIM = 200

#Vectorize the text samples into a S2 integer tensor
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(sentence_lines)
sequences = tokenizer_obj.texts_to_sequences(sentence_lines)

print(colored(sequences,'green'))

for x in range(len(sequences)):
	sequences[x] = np.append(sequences[x], stv[x])

# print(colored(sequences,'green'))

#define vocabulary size
vocab_size = len(tokenizer_obj.word_index) + 1

print(colored(sequences,'green'))

#pad sequences
word_index = tokenizer_obj.word_index
max_length = 9

triple_pad = pad_sequences(sequences, maxlen=max_length)
truth = df['truth'].values
print('Shape of triple tensor: ', triple_pad.shape)
print('Shape of truth tensor: ', truth.shape)

#map embeddings from loaded word2vec model for each word to the tokenizer_obj.word_index vocabulary & create a wordvector matrix

num_words = len(word_index)+1

print(colored(num_words,'cyan'))

model = Sequential()
model.add(layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model.add(layers.Conv1D(128, 4, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

#Split the data into training set and validation set
VALIDATION_SPLIT = 0.3

indices = np.arange(triple_pad.shape[0])
np.random.shuffle(indices)
triple_pad = triple_pad[indices]
truth = truth[indices]
num_validation_samples = int(VALIDATION_SPLIT * triple_pad.shape[0])

X_train_pad = triple_pad[:-num_validation_samples]
y_train = truth[:-num_validation_samples]
X_test_pad = triple_pad[-num_validation_samples:]
y_test = truth[-num_validation_samples:]

print('Shape of X_train_pad tensor: ',X_train_pad.shape)
print('Shape of y_train tensor: ',y_train.shape)
print('Shape of X_test_pad tensor: ',X_test_pad.shape)
print('Shape of y_test tensor: ',y_test.shape)

print(colored('Training...','green'))

history = model.fit(X_train_pad, y_train, batch_size=128, epochs=25, validation_data=(X_test_pad, y_test), verbose=2)

y_pred = model.predict_classes(x=X_test_pad)
metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print()
print(colored("Precision: ",'green'),colored(metrics[0],'blue'))
print(colored("Recall: ",'green'),colored(metrics[1],'blue'))
print(colored("F1: ",'green'),colored(metrics[2],'blue'))


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