import gensim
from corpus import Corpus
from document import Document
from typing import List, Dict, Tuple
from nltk.corpus import wordnet

reduce_for_near_words = 0.5
reduce_for_synonims = 0.5

def expand_query(doc_query : Document, corpus : Corpus):
    near_words = get_near_words(doc_query, corpus.words_id_dictionary, corpus.co_ocurrence_matrix) # La palabra mas cercana a todas las palabras de la query
    doc_query.near_words = set(near_words)
    for i in range (len(near_words)):
        doc_query.tags.append("")
    doc_query.data.extend(near_words)
    freq_of_repeated_words = {}
    expanded_query = get_words_from_synsets(get_hypernyms(get_synsets(doc_query, freq_of_repeated_words)))
    expanded_query = [[x for x in l if x in corpus.vocabulary] for l in expanded_query] # En esta linea nos despojamos de las palabras que no estan en el vocabulario
    doc_query.data = to_single_list(expanded_query)
    doc_query.to_doc2bow(corpus.words_id_dictionary)
    doc_query.data = increase_frequencies(doc_query.data, freq_of_repeated_words, corpus.words_id_dictionary)
    doc_query.to_tfidf_vector(corpus.tfidf_model)
    dict_of_tfidfs = reduce_tfidf_for_expanded_words(doc_query, corpus.words_id_dictionary)
    expanded_query = [[corpus.words_id_dictionary.token2id[word] for word in word_list] for word_list in expanded_query]
    doc_query.data = [[(id_word, dict_of_tfidfs[id_word]) for id_word in word_list] for word_list in expanded_query]

def reduce_tfidf_for_expanded_words(doc_query : Document, words_id_dictionary : gensim.corpora.Dictionary):
    dict_of_tfidfs = {}
    for word_id, tfidf in doc_query.data:
        if words_id_dictionary[word_id] in doc_query.near_words:
            tfidf = tfidf*reduce_for_near_words
        else:
            if not(words_id_dictionary.id2token[word_id] in doc_query.original):
                tfidf = tfidf * reduce_for_synonims
        dict_of_tfidfs[word_id] = tfidf
    return dict_of_tfidfs


def to_single_list(big_list : List[List]):
    new_list = []
    for l in big_list:
        new_list = new_list + l
    return new_list

def increase_frequencies(data : List[Tuple], dict_of_increases : Dict, words_id_dictionary : gensim.corpora.Dictionary):
    dict_of_increases = {words_id_dictionary.token2id[key] : value for key, value in dict_of_increases}
    return [(tpl[0], tpl[1] + dict_of_increases.get(tpl[0], 0)) for tpl in data]

def get_near_words(query : Document, dictionary : gensim.corpora.Dictionary, matrix, n = 1) -> List[str]:
    words_scores = []
    for word in dictionary.token2id:
        score = 0
        for qword in query.data:
            key = tuple(sorted([dictionary.token2id[word],dictionary.token2id[qword]]))
            if key in matrix.keys():
                score += matrix[key]
        words_scores.append((word,score))
    
    return [w[0] for w in sorted(words_scores,key=lambda word_score: word_score[1],reverse=True)][:n]

pos_tag_map = {
    'NOUN': [ wordnet.NOUN ],
    'ADJ': [ wordnet.ADJ, wordnet.ADJ_SAT ],
    'ADV': [ wordnet.ADV ],
    'VERB': [ wordnet.VERB ]
}

def get_synsets(doc_query : Document, freq_of_repeated_words : Dict):
    synsets = []
    freq_of_repeated_words = {}
    for i in range(len(doc_query.data)):
        if doc_query.data[i] in freq_of_repeated_words:
            freq_of_repeated_words[doc_query.data[i]] += 1
        freq_of_repeated_words[doc_query.data[i]] = 0
        synsets.append([doc_query.data[i]])
        if doc_query.tags[i] in pos_tag_map:
            synsets[i].extend(wordnet.synsets(doc_query.data[i], pos_tag_map[doc_query.tags[i]]))
    return synsets

def get_hypernyms(synsets):
    for synset in synsets:
        hypers = []
        for i in range(1,len(synset)):
            hypers.extend(synset[i].hypernyms())
        synset.extend(hypers)
    return synsets

def get_words_from_synsets(synsets):
    tokens = []
    for synset in synsets:
        tokens.append([synset[0]])
        for syn in synset[1:]:
            w = syn.name().split('.')[0]
            if w not in tokens[-1]:
                tokens[len(tokens)-1].append(w)
    return tokens