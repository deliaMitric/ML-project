from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

text = "NLTK is a powerful library for natural language processing."
words = word_tokenize(text)
sentences = sent_tokenize(text)

print(words)
print(sentences)

stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word.lower() not in stop_words]

print(filtered_words)
print(stop_words)
