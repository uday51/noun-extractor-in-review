import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.probability import FreqDist

review="The product is amazing! It works perfectly and I love it. However, the delivery was late, and the packaging was damaged. Overall, I would still recommend it.The product is amazing! It works perfectly and I love it. However, the delivery was late, and the packaging was damaged. Overall, I would still recommend it."

#converting sentence to lower
review=review.lower()

#Sentence tokenization
sentences=sent_tokenize(review)


#word tokenization
words=[word_tokenize(sentence) for sentence in sentences]


#flattening words
words=[word for sublist in words for word in sublist]


#Removing stop words
stop_words=set(stopwords.words("english"))
words=[word for word in words if word not in stop_words]


#Pos Tagging
pos_tags=pos_tag(words)


#Extracting nouns
nouns=[word for word, tag in pos_tags if tag in['NN','NNP','NNPS','NNS']]


#Count noun frequency
freq_dict=FreqDist(nouns)


#Get the most frequent nouns (top 2)
most_common_nouns=freq_dict.most_common(2)
print(dict(most_common_nouns))

