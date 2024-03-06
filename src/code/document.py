import spacy
import gensim
import ir_datasets
import os
import json
import nltk
from typing import Set, Dict, Tuple
from spacy.tokens import Token

nlp = spacy.load("en_core_web_sm")
Token.set_extension("sentence_number", default = 0)

class Document:
    def __init__(self, id : int, title : str, data, words_tfidf_dict = None, is_query = False, authors = "User", bib = "None"):
        self.title = title
        self.data = data
        self.id = id
        if words_tfidf_dict != None:
            self.words_tfidf_dict = words_tfidf_dict
        self.is_query = is_query
        self.authors = authors
    
    def process(self):
        self.find_authors()
        self.tokenize()
        self.remove_noise()
        self.remove_stop_words()
        self.create_word_sentence_list()
        if self.is_query:
            self.get_tags()
        self.morphological_reduce() #valorar si quitarlo o no

    def tokenize(self):
        self.data = [token for token in nlp(self.data)]
        sentence = 0
        for token in self.data:
            if token.text == '.':
                sentence = sentence + 1
            else:
                token._.sentence_number = sentence 

    def remove_noise(self):
        self.data = [token for token in self.data if token.is_alpha]
    
    def remove_stop_words(self):
        stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.data = [token for token in self.data if token.text not in stop_words]
    
    def create_word_sentence_list(self):
        self.word_sentence_list = [token._.sentence_number for token in self.data]
    
    def get_tags(self):
        self.tags = [token.pos_ for token in self.data]
    
    def morphological_reduce(self):
        self.data = [token.lemma_ for token in self.data]
    
    def to_doc2bow(self, words_id_dictionary : gensim.corpora.Dictionary):
        self.data = words_id_dictionary.doc2bow(self.data)
    
    def to_tfidf_vector(self, tfidf_model : gensim.models.TfidfModel):
        self.data = tfidf_model[self.data]
    
    def create_words_tfidf_dict(self):
        self.words_tfidf_dict = {}
        for word_id, tfidf in self.data:
            self.words_tfidf_dict[word_id] = tfidf
    
    def filter_unknown_words(self, vocabulary : Set[str]):
        self.data = [word for word in self.data if word in vocabulary]
    
    def save_original_words(self):
        self.original = set(self.data)

    def find_authors(self):
        tokenized_authors = [author.text for author in nlp(self.authors)]
        if "and" not in tokenized_authors:
            self.authors = [self.authors]
            return
        self.authors = extract_authors(tokenized_authors)

def extract_authors(tokenized_authors):
    all_authors = []
    current_author = ""
    last_token = ""
    for i in range(len(tokenized_authors)):
        if tokenized_authors[i] == "and":
            all_authors.append(current_author)
            all_authors.append(''.join(tokenized_authors[i+1:]))
            break
        if is_name(tokenized_authors[i]):
            if(last_token != ""):
                if(last_token == ','):
                    all_authors.append(current_author[-1])
                    current_author = ""
                elif (last_token != '-'):
                    current_author += " "
        current_author += tokenized_authors[i]
        last_token += tokenized_authors[i]
    return all_authors



def is_name(token):
    if (',' in token) | ('.' in token):
        return False
    return len(token)>=2