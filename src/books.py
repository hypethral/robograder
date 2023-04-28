import re

data = open('books.txt').read()
book = re.findall('[a-z]+', data.lower())

words_without_duplicates = []

for word in book:
  if word not in words_without_duplicates:
    words_without_duplicates.append(word)

output_sentence = " ".join(words_without_duplicates)


with open('myBook.txt', 'w') as f:
  f.write(output_sentence)