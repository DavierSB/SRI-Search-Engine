import json
from corpus import Corpus
import os
def get_recomendation(corpus : Corpus):
    documents = [[doc,0] for doc in corpus.document_vectors]
    for doc in documents:
        for author in doc[0].authors:
            doc[1] += corpus.authors_scores.get(author, 0)
        doc[1] /= max(1,len(doc[0].authors))
    return sorted(documents,key=lambda item: item[1],reverse=True)[:10]

def update_recomendation(corpus : Corpus, ranked_documents):
    for doc in ranked_documents:
        for author in doc[1].authors:
            corpus.authors_scores[author] = corpus.authors_scores.get(author,0) + doc[0]
    m = max(corpus.authors_scores.values())
    for key in corpus.authors_scores.keys():
        corpus.authors_scores[key] /= m
    save_authors(corpus)

def save_authors(corpus : Corpus):
    with open(os.getcwd() + '/data/authors.json','w') as out_f:
        json.dump(corpus.authors_scores, out_f)