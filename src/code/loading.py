from document import Document
from typing import List, Dict, Tuple, Set
import json
import gensim
import os
def load_document_vectors() -> List[Document]:
    with open(os.getcwd() + '/data/documents.json', 'r') as in_f:
        documents_dicts = json.load(in_f)
    return [Document(doc["id"], doc["title"], doc["data"], words_tfidf_dict = doc["words_tfidf_dict"], authors = doc["authors"]) for doc in documents_dicts]
def load_tfidf_model() -> Dict:
    return gensim.corpora.Dictionary.load(os.getcwd() + '/data/tfidf_model')
def load_words_id_dictionary() -> gensim.corpora.Dictionary:
    return gensim.corpora.Dictionary.load(os.getcwd() + '/data/words_id')
def load_co_ocurrence_matrix() -> Dict[Tuple, int]:
    with open(os.getcwd() + '/data/co_ocurrence_matrix.json') as in_f:
        co_ocurrence_matrix = {tuple(key): value for (key, value) in json.load(in_f)}
    return co_ocurrence_matrix
def load_authors_scores() -> Dict[str, int]:
    with open(os.getcwd() + '/data/authors.json') as in_f:
        return json.load(in_f)