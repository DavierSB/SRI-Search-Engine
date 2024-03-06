#from dataset import *
from document import Document
from typing import List, Dict, Tuple, Set
import ir_datasets
import gensim
import os
import json

def build_dataset(dataset_name):
    documents = get_documents(dataset_name)
    for doc in documents:
        doc.process()
    words_id_dictionary = build_words_id_dictionary(documents)
    co_ocurrence_matrix = build_co_ocurrence_matrix(documents, words_id_dictionary)
    for doc in documents:
        doc.to_doc2bow(words_id_dictionary)
    calculate_tf_idf(documents)
    for doc in documents:
        doc.create_words_tfidf_dict()
    authors_scores = build_authors_scores(documents)
    save(documents, co_ocurrence_matrix, authors_scores)

def get_documents(dataset_name : str):
    dataset = ir_datasets.load(dataset_name)
    return [Document(doc.doc_id, doc.title, doc.text, authors= doc.author) for doc in dataset.docs_iter()]

#en este momento tenemos las palabras no stopwords, lematizadas
def build_words_id_dictionary(dataset : List[str]) -> gensim.corpora.Dictionary:
    words_id_dictionary = gensim.corpora.Dictionary([doc.data for doc in dataset])
    words_id_dictionary.save(os.getcwd() + "/data/words_id")
    return words_id_dictionary

def build_authors_scores(documents : list[Document]):
    all_authors = dict()
    for doc in documents:
        for author in doc.authors:
            all_authors[author] = 0
    return all_authors

#en este momento tenemos en doc.data el BoW. Devolvemos los idf's
def calculate_tf_idf(documents : List[Document]) -> Dict:
    corpus = [doc.data for doc in documents]
    tfidf_model = gensim.models.TfidfModel(corpus, normalize = True)
    tfidf_model.save(os.getcwd() + '/data/tfidf_model')
    for doc in documents:
        doc.to_tfidf_vector(tfidf_model)

def build_co_ocurrence_matrix(documents : List[Document], dictionary : gensim.corpora.Dictionary, window_size = 2) -> Dict[Tuple, int]:
    matrix = {}
    for doc in documents:
        for i,token in enumerate(doc.data):
            start = max(0, i - window_size)
            end = min(len(doc.data), i + window_size + 1)
            for j in range(start,end):
                if (j == i) or (doc.word_sentence_list[i] != doc.word_sentence_list[j]):
                    continue
                key = tuple(sorted([dictionary.token2id[doc.data[j]], dictionary.token2id[token]]))
                matrix[key] = matrix.get(key, 0) + 0.5
    return matrix

def save(documents : List[Document], co_ocurrences_matrix : Dict[Tuple, int], authors_scores : Set[str]):
    documents_dicts = [doc.__dict__ for doc in documents]
    with open(os.getcwd() + '/data/documents.json', 'w') as out_f:
        json.dump(documents_dicts, out_f)
    with open(os.getcwd() + '/data/co_ocurrence_matrix.json', 'w') as out_f:
        json.dump(list(co_ocurrences_matrix.items()), out_f)
    with open('./data/authors.json','w') as out_f:
        json.dump(authors_scores, out_f)