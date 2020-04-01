from markov import models as markov_models

margo_txt = open("data/poor_margo.txt").read()
margo = markov_models.LM(gram=markov_models.WORD, n=5).train(margo_txt[30:-70].split())
margo.generate_text(100)
text = "And they went to"
while text.split()[-1] != ".":
    word = input("Write a word:")
    text += " " + word
    text += " " + margo.generate_next_gram(text)
    print (text)
