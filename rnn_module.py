from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import os
import json
import numpy as np
import pandas as pd
import re
import time
from glob import glob
import pickle
from os import listdir
import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GRU, LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.preprocessing import sequence
import nltk
nltk.download('punkt')
from absl import flags
from absl import logging
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_string("datafile", None, "Data to train (train+val+dev).")
flags.DEFINE_string("txt_col_name", "TEXT", "The name of the column storing the TEXT.")
flags.DEFINE_string("lbl_col_name", "LABEL", "The name of the column storing the LABEL.")

def preprocess(text, start="_start_", end="_end_", sentence_token="_sentence_token_"):
	if sentence_token is not None:
		sentences = nltk.tokenize.sent_tokenize(text)
		sentences = [s for s in sentences if len(s)>5]
		text = sentence_token.join(sentences)
	text = text.lower()
	return text

# define the captioning model
def define_model(vocab_size, max_length, loss="mse"):
	# embedding
	inputs = Input(shape=(max_length,))
	emb = Embedding(vocab_size, 200, mask_zero=True)(inputs)
	rnn = GRU(128, return_sequences=False)(emb)  
	fnn = Dense(1, activation='sigmoid')(rnn)
	model = Model(inputs=[inputs], outputs=fnn)
	model.compile(loss=loss, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
	# loss could also be e.g. nltk.translate.bleu_score.sentence_bleu
	print(model.summary())
	#plot_model(model, show_shapes=True, to_file='plot.png')
	return model

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

def read_data(csv_path):
	corpus = pd.read_csv(csv_path)
	train_pd, val_pd, dev_pd = prepare_data(corpus)
	return train_pd, val_pd, dev_pd

def set_tokenizer(corpus, txtname="TEXT"):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(corpus[txtname].to_numpy())
	return tokenizer

def prepare_data(corpus, split=0.8):
	# shuffle
	#data = data.sample(frac=1).reset_index(drop=True)
	data = shuffle(corpus)
	train_data = data[:int(split*data.shape[0])]
	eval_data = data[int(split*data.shape[0]):]
	val_data = eval_data[:int(0.5*eval_data.shape[0])] 
	dev_data = eval_data[int(0.5*eval_data.shape[0]):]
	return train_data, val_data, dev_data

def main(argv):
	logging.info(f"Training RNN on {FLAGS.datafile}")
	train_pd, val_pd, dev_pd = read_data(FLAGS.datafile)
	tokenizer = set_tokenizer(train_pd, FLAGS.txt_col_name)
	vocab_size = len(tokenizer.word_index) + 1
	print('Vocabulary Size: %d' % vocab_size)
	# define the experiment
	verbose, batch_size, n_epochs, max_length = 1, 128, 100, 512
	model_name = f'rnn.e{n_epochs}.len{max_length}'
	rnn_model = define_model(vocab_size, max_length)
	# prepare the dataset
	X, Y = tokenizer.texts_to_sequences(train_pd[FLAGS.txt_col_name].to_numpy()), train_pd[FLAGS.lbl_col_name].to_numpy()
	VX, VY = tokenizer.texts_to_sequences(dev_pd[FLAGS.txt_col_name].to_numpy()), dev_pd[FLAGS.lbl_col_name].to_numpy()
	X, VX = sequence.pad_sequences(X, maxlen=max_length), sequence.pad_sequences(VX, maxlen=max_length) # padding
	rnn_model.fit(X,Y,validation_data=(VX, VY), epochs=n_epochs, batch_size=batch_size, verbose=verbose)

if __name__ == "__main__": 
	app.run(main)
