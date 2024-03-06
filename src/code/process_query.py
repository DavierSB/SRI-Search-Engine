from document import Document
from corpus import Corpus
from gensim import similarities
from typing import List, Tuple
from random import randint
from query_expansion import expand_query
import boolean_model
import math

def process_query_with_extended_boolean_model(query : str, corpus : Corpus):
    doc_query = Document(None,'query',query, is_query= True)
    doc_query.process()
    doc_query.filter_unknown_words(corpus.vocabulary)
    doc_query.save_original_words()
    expand_query(doc_query, corpus)
    return doc_query


def search_documents(query : Document, corpus : Corpus):
    ranked_documents = []
    for doc in corpus.document_vectors:
        ranked_documents.append((sim(query, doc), doc))
    ranked_documents.sort(key= lambda tuple : tuple[0], reverse = True)
    return relevant_documents(ranked_documents), ranked_documents

def sim(query : Document, document : Document, p = 2):
    sum = 0
    for sub_query in query.data:
        sum += (1 - math.pow(1 - sim_or(sub_query, document), p))
    return pow(sum/max(len(query.data), 1), 1/p)

def sim_or(sub_query : List[Tuple], document : Document, p = 2):
    sum = 0
    div = 0
    for word_id, tfidf in sub_query:
        if str(word_id) in document.words_tfidf_dict:
            sum += math.pow(tfidf*document.words_tfidf_dict[str(word_id)],p)
            div += tfidf
    return pow(sum/max(div,1),1/p)

def relevant_documents(documents : List[Tuple]): # similaridad, Document
    if documents[0][0] < 0.05:
        return []
    answer = [documents[0]]
    last_score = documents[0][0]
    for doc in documents[1:]:
        score = doc[0]
        if score < 0.85*last_score:
            return answer
        answer.append(doc)
        if len(answer) == 5:
            break
        last_score = score
    return answer




def process_query_with_cosine_similarity(query : str, corpus : Corpus) -> List[int]:    
    doc_query = Document(None,'query',query)
    doc_query.process()
    doc_query.filter_unknown_words(corpus.vocabulary)
    doc_query.to_doc2bow(corpus.words_id_dictionary)
    doc_query.to_tfidf_vector(corpus.tfidf_model)

    index = similarities.SparseMatrixSimilarity([doc.data for doc in corpus.document_vectors], len(corpus.words_id_dictionary))
    similarity_scores = index[doc_query.data]
    scores = []
    for i, score in enumerate(similarity_scores):
        scores.append((score, i))
    scores.sort(reverse= True)
    answer_ids = []
    last_score = 0
    for score, id in scores:
        if len(answer_ids) == 0:
            answer_ids.append(id)
        if score < 0.85*last_score:
            return answer_ids
        last_score = score
        if score < 0.10:
            return answer_ids





def process_query_with_and_boolean_model(query : str, corpus : Corpus):
    return boolean_model.process_query_with_boolean_model(query, corpus.words_id_dictionary, "and")

def process_query_with_or_boolean_model(query : str, corpus : Corpus):
    return boolean_model.process_query_with_boolean_model(query, corpus.words_id_dictionary, "or")