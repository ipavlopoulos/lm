import pandas as pd
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
from sklearn.metrics import roc_auc_score
from tensorflow.keras import backend as K
!pip install bert-tensorflow
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

def create_tokenizer_from_hub_module(model_path):
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(model_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

class BERT(tf.layers.Layer):

    def __init__(self, n_fine_tune_layers=-1, output_representation='pooled_output', **kwargs):
        """
        :param output_representation: 'pooled_output' for CLS toke or 'sequence_output' for one-to-one outputs
        :n_fine_tune_layers: 0 for all, -n for top n layers trainable
        """
        assert output_representation in {"pooled_output", "sequence_output"}
        self.bert = None
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_representation = output_representation
        super(BERT, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(BERT_MODEL_PATH, trainable=self.trainable, 
                               name="{}_module".format(self.name))
        # Remove unused layers and set trainable parameters
        trainable_vars = [var for var in self.bert.variables 
                          if not "/cls/" in var.name 
                          and not "/pooler/" in var.name]

        # Select how many layers to fine tune
        if self.n_fine_tune_layers > 0:
          trainable_vars = trainable_vars[-self.n_fine_tune_layers :]

        # Add to trainable weights
        for var in trainable_vars:
          self._trainable_weights.append(var)
          
        # Update non-trainable weights
        for var in self.bert.variables:
          if var not in self._trainable_weights:
            self._non_trainable_weights.append(var)
          
        super(BERT, self).build(input_shape)

    def call(self, inputs):
        input_ids, input_mask, segment_ids = [K.cast(x, dtype="int32") for x in inputs]
        inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
        out = self.bert(inputs=inputs, as_dict=True, signature='tokens')
        # return out['sequence_output'][:, 0, :]
        return out[self.output_representation]

def build_model(trainable_layers, max_seq_length, output_representation="pooled_output", show_summary=False): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]
    
    bert_output = BERT(output_representation=output_representation, n_fine_tune_layers=trainable_layers)(bert_inputs)
    if output_representation == "pooled_output":
      dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    elif output_representation == "sequence_output":
      dense = tf.keras.layers.GRU(128)(bert_output)
    pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)
    
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    if show_summary: 
      model.summary()
    
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)
    
def get_features(features):
  input_ids, input_masks, segment_ids, labels = [], [], [], []
  for f in features:
    input_ids.append(f.input_ids)
    input_masks.append(f.input_mask)
    segment_ids.append(f.segment_ids)
    labels.append(f.label_id)
  return (
    np.array(input_ids),
    np.array(input_masks),
    np.array(segment_ids),
    np.array(labels).reshape(-1, 1),
  )

# get the fold shapes
def train_model(train, val, dev, tokenizer, DATA_COLUMN="TEXT", LABEL_COLUMN="LABEL", DATA2_COLUMN=None, tlayers=3, max_seq_length=128):
	print (index, train.shape, val.shape, dev.shape)
	# Use the InputExample class from BERT's run_classifier code to create examples from the data
	train_input = train.apply(lambda x: bert.run_classifier.InputExample(guid=None,
	                                                                   text_a = x[DATA_COLUMN], 
	                                                                   text_b = x[DATA2_COLUMN] if DATA2_COLUMN else None, 
	                                                                   label = x[LABEL_COLUMN]), axis = 1)
	dev_input = dev.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
	                                                                   text_a = x[DATA_COLUMN], 
	                                                                   text_b = x[DATA2_COLUMN] if DATA2_COLUMN else None, 
	                                                                   label = x[LABEL_COLUMN]), axis = 1)
	val_input = val.apply(lambda x: bert.run_classifier.InputExample(guid=None,
	                                                                   text_a = x[DATA_COLUMN], 
	                                                                   text_b = x[DATA2_COLUMN] if DATA2_COLUMN else None, 
	                                                                   label = x[LABEL_COLUMN]), axis = 1)
	# get the features
	train_features = bert.run_classifier.convert_examples_to_features(train_input, label_list, max_seq_length, tokenizer)
	dev_features = bert.run_classifier.convert_examples_to_features(dev_input, label_list, max_seq_length, tokenizer)
	val_features = bert.run_classifier.convert_examples_to_features(val_input, label_list, max_seq_length, tokenizer)

	# adjust to BERT  
	train_input_ids, train_input_masks, train_segment_ids, train_labels = get_features(train_features)
	dev_input_ids, dev_input_masks, dev_segment_ids, dev_labels = get_features(dev_features)
	val_input_ids, val_input_masks, val_segment_ids, val_labels = get_features(val_features)

	# build the model
	model = build_model(trainable_layers=tlayers, max_seq_length=max_seq_length, output_representation="pooled_output")

	# Early stopping
	stopping = EarlyStopping(monitor='val_loss',patience=3, verbose=0, min_delta=0.01, restore_best_weights=True, mode="min")

	# Instantiate variables
	initialize_vars(sess)

	model.fit(
	    [train_input_ids, train_input_masks, train_segment_ids], 
	    train_labels,
	    validation_data=([val_input_ids, val_input_masks, val_segment_ids], val_labels),
	    epochs=20,
	    callbacks = [stopping],
	    batch_size=32
	)

	# evaluate
	predictions = model.predict([dev_input_ids, dev_input_masks, dev_segment_ids]) # predictions before we clear and reload model
	auc = roc_auc_score(dev_labels, predictions)
	print('ROC AUC: {:.4f}'.format(auc))
	print ('Best epoch: ', stopping.stopped_epoch)
	return model

def __main__():
	sess = tf.Session() # Initialize session
	BERT_MODEL_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1" # path to an uncased (all lowercase) version of BERT
	max_seq_length = 400
	!wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip && unzip cased_L-12_H-768_A-12.zip
	FLAGS = tf.app.flags.FLAGS
	tf.app.flags.DEFINE_integer("train", None, "The Dataset in CSV format. It should have a TEXT column and a LABEL column. Any other text column (e.g., for BERT) should be named CONTEXT")	
	tf.app.flags.DEFINE_integer("dev", None, "The Dataset in CSV format. It should have a TEXT column and a LABEL column. Any other text column (e.g., for BERT) should be named CONTEXT")	
	tf.app.flags.DEFINE_integer("val", None, "The Dataset in CSV format. It should have a TEXT column and a LABEL column. Any other text column (e.g., for BERT) should be named CONTEXT")	
	tokenizer = create_tokenizer_from_hub_module(BERT_MODEL_PATH)
	# get the data	
	train_data = pd.read_csv(FLAGS.train)
	val_data = pd.read_csv(FLAGS.val)
	dev_data = pd.read_csv(FLAGS.dev)
	train_model(train_data, val_data, dev_data, tokenizer)
