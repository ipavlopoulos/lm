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
>>> cml.bpg('this an example this is an example')
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
>>> wlm.ppl("this is an example")
29.897352853986263
```

### Following...
* Neural Language Modeling (experimental implementation included)
* Text Classification
