# lm
Language Modelling (LM) modules

# How to
```
git clone https://github.com/ipavlopoulos/lm.git
```
Let's assume you have a pandas load with data, including the labels "TEXT" (raw) and "LABEL" (0/1). Then, it goes like this:
```
>>> from lm import attention_rnn_module as RNN, rnn_module as RNN
>>> import keras
>>> train_pd, val_pd, dev_pd = RNN.prepare_data(dataset_pd)
>>> tokenizer = RNN.Tokenizer()
>>> tokenizer.fit_on_texts(train["TEXT"].to_numpy())
>>> vocab_size = len(tokenizer.word_index) + 1
>>> verbose, batch_size, n_epochs, max_length = 1, 128, 100, 400
>>> rnn_model = RNN.define_model(vocab_size, max_length)
>>> X, Y = tokenizer.texts_to_sequences(train["TEXT"].to_numpy()), train["LABEL"].to_numpy()
>>> DX, DY = tokenizer.texts_to_sequences(dev["TEXT"].to_numpy()), dev["LABEL"].to_numpy()
>>> X, DX = RNN.sequence.pad_sequences(X, maxlen=max_length), RNN.sequence.pad_sequences(DX, maxlen=max_length) # padding
>>> early = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.001, patience=3, verbose=1, mode='auto', restore_best_weights=True)
>>> model.fit(X,Y,validation_data=(DX, DY), epochs=n_epochs, batch_size=batch_size, verbose=verbose, callbacks=[early])
