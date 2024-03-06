import ir_datasets
import nltk
import spacy
import gensim

def initialize():
    documents = load_dataset("cranfield")
    tokenized_docs = morphological_reduction_spacy(remove_stopwords_spacy(remove_noise_spacy(tokenization_spacy(documents))), True)
    dictionary = get_dictionary(tokenized_docs)
    tokenized_docs = filter_tokens_by_occurrence(tokenized_docs,dictionary)
    document_representation = get_document_representation(tokenized_docs, dictionary, [],True)
    word_representation = get_word_representation(document_representation,dictionary)
    return word_representation

def process_query_with_boolean_model(query : str, dictionary : gensim.corpora.Dictionary, implementation : str):
    procesed_query = process_query(query,dictionary)
    if implementation == "and":
       return and_implementation(procesed_query, word_representation)
    return or_implementation(procesed_query, word_representation)

def process_query(query,dictionary):
    proceced_query = morphological_reduction_spacy(remove_stopwords_spacy(remove_noise_spacy(tokenization_spacy([(0,query)]))), True)
    proceced_query = filter_tokens_by_occurrence(proceced_query,dictionary)
    return [id for (id,sec) in get_document_representation(proceced_query, dictionary, [],True)[0][1]]

def and_implementation(query,word_representation):
    if len(query) == 0:
        return []
    result = word_representation[query[0]]
    for id in query[1:]:
        for doc_id in result:
            if doc_id not in word_representation.get(id, []):
                result.remove(doc_id)
    return result

def or_implementation(query,word_representation):
    if len(query) == 0:
        return []
    result = set()
    for id in query:
        for doc_id in word_representation.get(id, []):
            if doc_id not in result:
                result.add(doc_id)
    return list(result)

nlp = spacy.load("en_core_web_sm")

def load_dataset(data_name):
    dataset = ir_datasets.load(data_name)
    return [(doc.doc_id,doc.text) for doc in dataset.docs_iter()]

def tokenization_spacy(texts):
  return [(id,[token for token in nlp(doc)]) for (id,doc) in texts]

def remove_noise_spacy(tokenized_docs):
  return [(id,[token for token in doc if token.is_alpha]) for id,doc in tokenized_docs]

def remove_stopwords_spacy(tokenized_docs):
  stopwords = spacy.lang.en.stop_words.STOP_WORDS
  return [
      (id,[token for token in doc if token.text not in stopwords]) for id,doc in tokenized_docs
  ]

def morphological_reduction_spacy(tokenized_docs, use_lemmatization=True):
  stemmer = nltk.stem.PorterStemmer()
  return [
    (id,[token.lemma_ if use_lemmatization else stemmer.stem(token.text) for token in doc])
    for id,doc in tokenized_docs
  ]

def get_dictionary(tokenized_docs,no_below=5, no_above=0.5):
  dictionary = gensim.corpora.Dictionary([doc for _,doc in tokenized_docs])
  dictionary.filter_extremes(no_below=no_below, no_above=no_above)
  return dictionary

def filter_tokens_by_occurrence(tokenized_docs,dictionary):
  
  filtered_words = [word for _, word in dictionary.iteritems()]
  filtered_tokens = [
      (id,[word for word in doc if word in filtered_words])
      for id,doc in tokenized_docs
  ]

  return filtered_tokens

def build_vocabulary(dictionary):
  vocabulary = list(dictionary.token2id.keys())
  return vocabulary

def get_document_representation(tokenized_docs, dictionary, vector_repr, use_bow=True):
    corpus = [(id,dictionary.doc2bow(doc)) for id,doc in tokenized_docs]

    if use_bow:
        vector_repr = corpus
    else:
        tfidf = gensim.models.TfidfModel(corpus)
        vector_repr = [tfidf[doc] for doc in corpus]

    return vector_repr

def get_word_representation(document_representation, dictionary):
    word_representation = {id_word:[] for id_word in dictionary}
    for id_doc, words in document_representation:
        for word in words:
          word_representation[word[0]].append(id_doc)
    return word_representation


word_representation = initialize()