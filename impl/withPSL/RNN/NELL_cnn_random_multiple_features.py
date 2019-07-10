import warnings
warnings.filterwarnings("ignore")
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from sklearn.metrics import precision_recall_fscore_support
from termcolor import colored
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense
from keras.layers import Embedding, Input, Conv1D, GlobalMaxPooling1D, concatenate


# define documents
df = pd.DataFrame()
df = pd.read_csv('../../data/triple_with_stv.csv')
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

#define vocabulary size
vocab_size = len(tokenizer_obj.word_index) + 1

# print(colored(sequences,'green'))

#pad sequences
word_index = tokenizer_obj.word_index
max_length = 5

triple_pad = pad_sequences(sequences, maxlen=max_length)
truth = df['truth'].values
print('Shape of triple tensor: ', triple_pad.shape)
print('Shape of truth tensor: ', truth.shape)

#map embeddings from loaded word2vec model for each word to the tokenizer_obj.word_index vocabulary & create a wordvector matrix

num_words = len(word_index)+1

print(colored(num_words,'cyan'))

# first input model
emb = Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length)
input_shape = triple_pad.shape
print(colored('Input SHAPE for sequences','cyan'))
# print(input_shape)
visible1 = Input(shape=input_shape)
conv11 = Conv1D(128, 4, activation='relu')(visible1)
pool11 = GlobalMaxPooling1D()(conv11)
den1 = Dense(10, activation='relu')(pool11)

# second input layer
input_shape_stv = np.array(stv).shape
print(colored("Input Shape for stv: ",'cyan'))
print(input_shape_stv)
visible2 = Input(shape=input_shape_stv)
den2 = Dense(10, activation='relu')(visible2)

# # merge input models
merge = concatenate([den1, den2])

# interpretation model
hidden1 = Dense(10, activation='relu')(merge)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=[visible1, visible2], outputs=output)

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

# X_train_pad = np.expand_dims(X_train_pad, 1)

history = model.fit([X_train_pad,np.array(stv[:-num_validation_samples])], y_train, batch_size=128, epochs=25, validation_data=([X_test_pad,np.array(stv[-num_validation_samples:])], y_test), verbose=2)

y_pred = model.predict_classes(x=[X_test_pad,np.array(stv[-num_validation_samples:])])
metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print()
print(colored("Precision: ",'green'),colored(metrics[0],'blue'))
print(colored("Recall: ",'green'),colored(metrics[1],'blue'))
print(colored("F1: ",'green'),colored(metrics[2],'blue'))
