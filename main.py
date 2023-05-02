import nltk
from gensim.models import  Word2Vec
import re
nltk.download('stopwords')

text = "Python is a high-level, interpreted programming language. It is used for web development, data analysis, artificial intelligence, and more."

stop_words = set(nltk.corpus.stopwords.words('english'))

words = nltk.word_tokenize(text)

print(words)

words = [word.lower() for word in words if re.match('\w+', word)]


words = [word for word in words if word not in stop_words]


model = Word2Vec([words], min_count=1, vector_size=50)

paraphrase = []

for word in words:
    paraphrase.append(model.wv.most_similar(word, topn=1)[0][0])

paraphrase_text =  ' '.join(paraphrase)

print(paraphrase_text)