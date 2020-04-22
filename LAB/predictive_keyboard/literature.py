from markov import models as markov_models

# read a book
book = open("DATA/book.txt").read()

# train a 3-gram LM
book_lm = markov_models.LM(gram=markov_models.WORD, n=5).train(book[30:-70].split())

# generate some text
book_lm.generate_text(100)

# generate text interactively
text = "And they went to"
while text.split()[-1] != ".":
    word = input("Write a word:")
    text += " " + word
    text += " " + book_lm.generate_next_gram(text)
    print(text)
