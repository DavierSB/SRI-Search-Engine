from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from corpus import Corpus
from process_query import process_query_with_cosine_similarity, process_query_with_extended_boolean_model, search_documents
from document import Document
import random
import ir_datasets
from statistics import mean
    
def evaluate(data_name = "cranfield", process_query_method = process_query_with_cosine_similarity):
    corpus = Corpus()
    dataset = ir_datasets.load(data_name)
    querys = dataset.queries_iter()
    evaluations_list = []
    for i,query in enumerate(querys):
        predicted_documents_ids = process_query_method(query.text,corpus)
        if isinstance(predicted_documents_ids, Document):
            predicted_documents_ids = search_documents(predicted_documents_ids, corpus)[0]
            predicted_documents_ids = [tpl[1].id for tpl in predicted_documents_ids]
        right_documents_id = get_right_documents(query.query_id,dataset)
        predicted_classification, right_classification = get_classification(corpus.document_vectors,predicted_documents_ids,right_documents_id)
        evaluations = get_stats(get_confusion_matrix(right_classification,predicted_classification))
        evaluations_list.append(evaluations)
    precisions, recovers, f1s, fallouts = zip(*evaluations_list)
    precisions = [p for p in precisions if p is not None]
    recovers = [r for r in recovers if r is not None]
    f1s = [f for f in f1s if f is not None]
    fallouts = [f for f in fallouts if f is not None]
    return (mean(precisions), mean(recovers), mean(f1s) if (len(f1s) > 0) else None, mean(fallouts))
    print("The average precisions is: " + str(mean(precisions)))
    print("The average recovered is: " + str(mean(recovers)))
    if len(f1s) > 0:
        print("The average f1 is: " + str(mean(f1s)))
    if len(fallouts) > 0:
        print("The average fallouts is: " + str(mean(fallouts)))

def get_right_documents(query_id,dataset):
    return (
    [
      doc_id
      for (queryt_id, doc_id, relevance, _) in dataset.qrels_iter()
      if queryt_id == query_id and relevance in [1, 2, 3, 4]
    ])

def get_classification(documents, predicted_documents_id, right_documents_id):
    predicted_classification = [0 for i in range(len(documents) + 1)]
    right_classification = [0 for i in range(len(documents) + 1)]
    for id in predicted_documents_id:
        predicted_classification[int(id)] = 1
    for id in right_documents_id:
        right_classification[int(id)] = 1
    return predicted_classification, right_classification

def get_confusion_matrix(right_classification,model_classification):
    lenth = len(right_classification)
    true_negative = 0 
    false_positive = 0 
    false_negative = 0
    true_positive = 0
    for i in range(0,lenth):
        if (right_classification[i] == 0) and (model_classification[i] == 0):
            true_negative += 1
        if (right_classification[i] == 0) and (model_classification[i] == 1):
            false_positive += 1
        if (right_classification[i] == 1) and (model_classification[i] == 0):
            false_negative += 1
        if (right_classification[i] == 1) and (model_classification[i] == 1):
            true_positive += 1
    return true_negative, false_positive, false_negative, true_positive
def get_stats(matrix):
    true_negative, false_positive, false_negative, true_positive = matrix
    predicted_positive = true_positive + false_positive
    right_positive = true_positive + false_negative
    right_negative = false_positive + true_negative
    precision = None
    recovered = None
    f1 = None
    fallout = None
    if predicted_positive != 0:
        precision = true_positive/predicted_positive
    if right_positive != 0:
        recovered = true_positive/right_positive
    if ((precision is not None) and (recovered is not None) and (precision + recovered > 0)):
        f1 = (2*precision*recovered)/(precision + recovered)
    if right_negative != 0:
        fallout = false_positive/right_negative
    return (precision, recovered, f1, fallout)