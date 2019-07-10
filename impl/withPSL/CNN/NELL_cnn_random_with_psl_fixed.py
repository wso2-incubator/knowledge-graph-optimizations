import string
import warnings

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.layers import GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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
triple_lines = list()
lines = df['triple'].values.tolist()
stv = df['stv'].values.reshape(-1, 1)

for line in lines:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    triple_lines.append(words)

print(colored(len(triple_lines), 'green'))

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
max_length = 9

triple_pad = pad_sequences(sequences, maxlen=max_length)
truth = df['truth'].values
print('Shape of triple tensor: ', triple_pad.shape)
print('Shape of truth tensor: ', truth.shape)

# map embeddings from loaded word2vec model for each word to the tokenizer_obj.word_index vocabulary & create a wordvector matrix

num_words = len(word_index) + 1

print(colored(num_words, 'cyan'))

input1 = layers.Input(shape=(max_length,))
embedding = layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length)(input1)

cov = layers.Conv1D(128, 4, activation='relu')(embedding)
pooling = layers.GlobalMaxPooling1D()(cov)

input2 = layers.Input(shape=(1,))

concat = layers.Concatenate(axis=-1)([pooling, input2])

l1 = layers.Dense(10, activation='relu')(concat)
out = layers.Dense(1, activation='sigmoid')(l1)

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

history = model.fit([X_train_pad, X_train_psl], y_train, batch_size=128, epochs=25,
                    validation_data=([X_test_pad, X_test_psl], y_test), verbose=2)

y_pred = (model.predict(x=[X_test_pad, X_test_psl]) > 0.5).astype(np.int32)
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
