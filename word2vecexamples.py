# pylint: skip-file
#example-1
from gensim.models import Word2Vec
from gensim.test.utils import common_texts

#train word2vec model on some sample sentences
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)

#get the vector representation of a word => common texts has about 29 words
print(common_texts)

commonword = 'human'
vector = model.wv[commonword]

#find similar words
similar_words = model.wv.most_similar(commonword)
print (f"vector rep of {commonword} : {vector}")
print(f"similar words to {commonword} : {similar_words}")

#example-2
import bs4 as bs
import urllib.request
import re
import nltk
from nltk.tokenize import sent_tokenize

scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')
article = scrapped_data .read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:
    article_text += p.text

# Cleaing the text
processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
processed_article = re.sub(r'\s+', ' ', processed_article)

# Ensure you have downloaded the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

print(processed_article)
# Preparing the dataset
all_sentences = sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]

#import gensim.models
from gensim.models import Word2Vec
word2vec = Word2Vec(all_words, min_count=1)

vocabulary = word2vec.wv.key_to_index
print(vocabulary)
