from tester import evaluate
from process_query import process_query_with_extended_boolean_model, process_query_with_or_boolean_model, process_query_with_and_boolean_model

print("Cosine Similarity")
evaluate()
print("______________________________________________________________")
print("Boolean Extended Model")
evaluate(process_query_method = process_query_with_extended_boolean_model)
print("______________________________________________________________")
print("Boolean Model using OR")
evaluate(process_query_method= process_query_with_or_boolean_model)
print("_______________________________________________________________")
print("Boolean Model using AND")
evaluate(process_query_method= process_query_with_and_boolean_model)