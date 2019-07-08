import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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

import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
# one hot encode

seed = 64
np.random.seed(seed)
tf.set_random_seed(seed)


df = pd.DataFrame()
df = pd.read_csv('../../data/yago_triples.csv')
triple_lines = list()
lines = df['triple'].values.tolist()

for line in lines:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('','',string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    triple_lines.append(words)

print(colored(len(triple_lines),'green'))
EMBEDDING_DIM = 200

#Vectorize the text samples into a S2 integer tensor
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(triple_lines)
sequences = tokenizer_obj.texts_to_sequences(triple_lines)

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

# Define Model
model = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            input_length=max_length,
                            trainable=True)

model.add(embedding_layer)
model.add(GRU(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(colored(model.summary(),'cyan'))

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


# import matplotlib.pyplot as plt
# plt.style.use('ggplot')
#
# def plot_history(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)
#
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#
# plt.show(plot_history(history))
