import streamlit as st
import pandas as pd
from corpus import Corpus
from process_query import process_query_with_boolean_extended_model, search_documents
from recomendation import update_recomendation, get_recomendation

st.title("Search Engine")
corpus = Corpus()
recomendation = get_recomendation(corpus)
query = st.text_input("Introduce your query:")
   
if query != "":
    query = process_query_with_boolean_extended_model(query,corpus)      
    founded_docs, _ = search_documents(query,corpus)
    id_search = [int(doc.id) for _,doc in founded_docs]
    title_search = [doc.title for _,doc in founded_docs]
    score_search = [score for score, _ in founded_docs]

    df = pd.DataFrame({'Id':id_search,'Title':title_search, 'Score':score_search})
    st.table(df.set_index(df.columns[0]))
    update_recomendation(corpus,founded_docs)
    recomendation = get_recomendation(corpus)

with st.expander("Recomendations"):
    recomendations_ids = [int(doc.id) for doc, _ in recomendation]
    recomendations_titles = [doc.title for doc, _ in recomendation]
    recomendation_dataframe = pd.DataFrame({'Id':recomendations_ids,'Title':recomendations_titles})
    st.table(recomendation_dataframe.set_index(recomendation_dataframe.columns[0]))
