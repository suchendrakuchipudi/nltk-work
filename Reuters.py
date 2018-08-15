# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:19:20 2018

@author: suchendra
"""

import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk import ne_chunk
#from nltk.probability import FreqDist
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

from nltk.corpus import reuters
def collection_stats():
    # List of documents
    documents = reuters.fileids()
    print(str(len(documents)) + " documents");
 
    train_docs = list(filter(lambda doc: doc.startswith("train"),
                        documents));
    print(str(len(train_docs)) + " total train documents");
 
    test_docs = list(filter(lambda doc: doc.startswith("test"),
                       documents));
    print(str(len(test_docs)) + " total test documents");
 
    # List of categories
    categories = reuters.categories();
    print(str(len(categories)) + " categories");
 
    # Documents in a category
    category_docs = reuters.fileids("acq");
 
    # Words for a document
    document_id = category_docs[0]
    document_words = reuters.words(category_docs[0]);
    print(document_words);
 
    # Raw document
    #print(reuters.raw(document_id));
 
#preprocessing the data by pos, named_entity, chunking and finally fikltering using regrex
def tokenize(text):
    words=map(lambda word: word.lower(), word_tokenize(text))
    words=[word for word in words if word not in cachedStopWords]
    words=list(map(lambda token: PorterStemmer().stem(token), words))
    tagged=nltk.pos_tag(words)
    words=nltk.ne_chunk(tagged) 
    chunkGram= r"""chunk: {<NN>+}
    }<VB.?|IN|DT|TO|VBZ|RP>+{"""
    chunkParser=nltk.RegexpParser(chunkGram)
    chunked=chunkParser.parse(words)
    return chunked


#training through the docs and returning the most popular 100 words to select them as features    
def find_features(train_docs):
    all_words=[]
    for i in range(len(train_docs)):
        text=reuters.raw(train_docs[i])
        words=map(lambda word: word.lower(), word_tokenize(text))
        words=[word for word in words if word not in cachedStopWords]
        words=list(map(lambda token: PorterStemmer().stem(token), words))
        for w in words:
            all_words.append(w.lower())
    all_words=FreqDist(all_words)
    common_words=list(all_words.keys())[:100]
    return common_words
        
    
#collection_stats()
category_docs = reuters.fileids("acq")
train_docs = list(filter(lambda doc: doc.startswith("train"),category_docs));
test_docs=list(filter(lambda doc: doc.startswith("test"),category_docs));
find_features(train_docs)




