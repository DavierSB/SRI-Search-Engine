from corpus import Corpus
from pre_processing import build_dataset
from process_query import process_query_with_cosine_similarity, process_query_with_extended_boolean_model, search_documents
from recomendation import update_recomendation, get_recomendation

def ejecution(already_preprocessed = True, process_query_method = process_query_with_cosine_similarity):
    if not already_preprocessed:
        print("Ejecution start")
        build_dataset("cranfield")
    corpus = Corpus()
    while True:
        print("Te sugerimos...")
        recommendations = get_recomendation(corpus)
        for doc, score in recommendations:
            print(str(doc.id) + " " + str(score))
        print("Introduce your query")
        query = input()
        query = process_query_method(query,corpus)
        result, ranked_documents = search_documents(query,corpus)
        update_recomendation(corpus,ranked_documents)
        for doc in result:
            print(doc[1].title, doc[0])
ejecution(True, process_query_method= process_query_with_extended_boolean_model)