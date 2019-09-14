import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import precision_recall_fscore_support
from termcolor import colored

warnings.filterwarnings("ignore")

seed = 64
np.random.seed(seed)
tf.set_random_seed(seed)

df = pd.DataFrame()
df = pd.read_csv('../../../data/yago.csv')
# df = pd.read_csv('../../../data/triple_with_stv.csv')
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
max_length = 5

triple_pad = np.array(sequences)

num_words = len(word_index) + 1

print(colored(num_words, 'cyan'))

dropout = [0.2,0.3,0.4]
activation = ['relu','sigmoid']
optimizer = ['adam']
losses = ['binary_crossentropy']
batch_size = [64, 128]
epochs = [12,25,50]

dropout_arr = np.array([])
activation_arr = np.array([])
optimizer_arr = np.array([])
losses_arr = np.array([])
batch_size_arr = np.array([])
epochs_arr = np.array([])
precision_arr = np.array([])
recall_arr = np.array([])
f1_arr = np.array([])


for drp in dropout:
	for act in activation:
		for optm in optimizer:
			for los in losses:
				for btsize in batch_size:
					for ep in epochs:

						input1 = layers.Input(shape=(max_length,))
						embedding = layers.Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length)(input1)

						cov = layers.Conv1D(128, 4, activation='relu')(embedding)
						pooling = layers.GlobalMaxPooling1D()(cov)

						input2 = layers.Input(shape=(1,))

						concat = layers.Concatenate(axis=-1)([pooling, input2])

						l1 = layers.Dense(10, activation=act)(concat)
						out = layers.Dense(1, activation='sigmoid')(l1)

						model = models.Model(inputs=[input1, input2], outputs=[out])
						model.compile(optimizer=optm, loss=los, metrics=['accuracy'])

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

						history = model.fit([X_train_pad, X_train_psl], y_train, batch_size=btsize, epochs=ep,
											validation_data=([X_test_pad, X_test_psl], y_test), verbose=2)

						y_pred = (model.predict(x=[X_test_pad, X_test_psl])>0.1).astype(np.int32)
						metrics = precision_recall_fscore_support(y_test, y_pred, average='weighted')


						dropout_arr = np.append(dropout_arr, drp)
						activation_arr = np.append(activation_arr, act)
						optimizer_arr = np.append(optimizer_arr, optm)
						losses_arr = np.append(losses_arr, los)
						batch_size_arr = np.append(batch_size_arr, btsize)
						epochs_arr = np.append(epochs_arr, ep)
						precision_arr = np.append(precision_arr, metrics[0])
						recall_arr = np.append(recall_arr,metrics[1])
						f1_arr = np.append(f1_arr,metrics[2])

print(colored("\n\n\nSUCCESSFULLY COMPLETED GRID SEARCH OPTIMIZATION", 'green'))

print(colored("Report", 'blue'))

print()
print("Number of test run: "+ str(len(f1_arr))+'\n')
print(colored("dropout\tactivation\toptimizer\tloss\tbatch size\tepochs\tprecision\trecall\tf1 score\n",'cyan'))
for x in range(len(f1_arr)):
	print(str(dropout_arr[x])+'\t'+ activation_arr[x]+'\t'+optimizer_arr[x]+'\t'+losses_arr[x]+'\t'
		+ str(batch_size_arr[x])+'\t'+ str(epochs_arr[x])+'\t'+str(precision_arr[x])+'\t'
			+ str(recall_arr[x])+'\t'+str(f1_arr[x]))