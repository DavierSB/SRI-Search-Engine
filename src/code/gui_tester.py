import streamlit as st
from tester import evaluate
from process_query import *

col1, col2 = st.columns(2)

col1.header("Boolean Model with OR")
boolean_or_model_precision, boolean_or_model_recover, boolean_or_model_f1, boolean_or_model_failout = evaluate(process_query_method= process_query_with_or_boolean_model) 
col1.text("Precision: " + str(boolean_or_model_precision))
col1.text("Recover: " + str(boolean_or_model_recover))
col1.text("f1: " + str(boolean_or_model_f1))
col1.text("Failout: " + str(boolean_or_model_failout))

col1.header("Vectorial Model       ")
vectorial_model_precision, vectorial_model_recover, vectorial_model_f1, vectorial_model_failout = evaluate() 
col1.text("Precision: " + str(vectorial_model_precision))
col1.text("Recover: " + str(vectorial_model_recover))
col1.text("f1: " + str(vectorial_model_f1))
col1.text("Failout: " + str(vectorial_model_failout))

col2.header("Boolean Model with AND")
boolean_and_model_precision, boolean_and_model_recover, boolean_and_model_f1, boolean_and_model_failout = evaluate(process_query_method= process_query_with_and_boolean_model) 
col2.text("Precision: " + str(boolean_and_model_precision))
col2.text("Recover: " + str(boolean_and_model_recover))
col2.text("f1: " + str(boolean_and_model_f1))
col2.text("Failout: " + str(boolean_and_model_failout))

col2.header("Extended Boolean Model")
boolean_extended_model_precision, boolean_extended_model_recover, boolean_extended_model_f1, boolean_extended_model_failout = evaluate(process_query_method= process_query_with_extended_boolean_model)
col2.text("Precision: " + str(boolean_extended_model_precision))
col2.text("Recover: " + str(boolean_extended_model_recover))
col2.text("f1: " + str(boolean_extended_model_f1))
col2.text("Failout: " + str(boolean_extended_model_failout))
