from loading import load_document_vectors, load_words_id_dictionary, load_co_ocurrence_matrix, load_tfidf_model, load_authors_scores
class Corpus:
    def __init__(self) -> None:
        self.document_vectors = load_document_vectors()
        self.words_id_dictionary = load_words_id_dictionary()
        self.co_ocurrence_matrix = load_co_ocurrence_matrix()
        self.tfidf_model = load_tfidf_model()
        self.vocabulary = self.words_id_dictionary.token2id.keys()
        self.authors_scores = load_authors_scores()