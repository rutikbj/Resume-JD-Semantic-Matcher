from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import spacy

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
spacy_trf = spacy.load('en_core_web_trf')
bert_model_advanced = SentenceTransformer('all-mpnet-base-v2')
 
def tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def bow_similarity(text1, text2):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

def bert_similarity(text1, text2):
    v1 = bert_model.encode(text1, convert_to_tensor=True)
    v2 = bert_model.encode(text2, convert_to_tensor=True)
    return util.cos_sim(v1, v2).item()

def bert_similarity_advanced(text1, text2):
    v1 = bert_model_advanced.encode(text1, convert_to_tensor=True)
    v2 = bert_model_advanced.encode(text2, convert_to_tensor=True)
    return util.cos_sim(v1, v2).item()

""" def spacy_doc_similarity(text1, text2):
    doc1 = spacy_trf(text1)
    doc2 = spacy_trf(text2)
    print("doc1.vector_norm:", doc1.vector_norm)
    print("doc2.vector_norm:", doc2.vector_norm)
    return doc1.similarity(doc2)"""

def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0