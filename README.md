# lm
Language Modelling (LM) modules.

### Character-based N-Gram model.
* Uses N-Gram statistics to generate the next character.
* It includes a Bits Per Character implementation for evaluation.
```python
>>> from markov.models import LM
>>> clm = LM(gram="CHAR").train(" ".join(["this", "is", "an", "example", "this", "an"]))
>>> clm.generate_text()
'this an example this is an example this an example this...'
>>> cml.cross_entropy('this an example this is an example')
0.058823529411764705
```
### Word-based N-Gram model.
* Uses N-Gram statistics to generate the next word.
* It includes a Perplexity implementation for evaluation.
```python
>>> from markov.models import LM
>>> wlm = LM(gram="WORD").train(["this", "is", "an", "example", "this", "an"])
>>> wlm.generate_text()
'this is an example this an example this example is is this this...'
>>> wlm.cross_entropy("this is an example".split())
22.423014640489697
```

### Neural
* Uses an RNN to generate the next word.
```python
>>> from neural import models
>>> rnn = models.RNN(epochs=1)
>>> data = """ Jack and Jill went up the hill\n
		To fetch a pail of water\n
		Jack fell down and broke his crown\n
		And Jill came tumbling after\n """
>>> rnn.train(data)
>>> rnn.cross_entropy("Jack fell down and broke his crown\n")
```
