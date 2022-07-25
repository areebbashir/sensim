# I will be using bert for getting embeddings for the two sentences and then simple cosine similarity is
# used to calculate the similarity between the sentences
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
model = SentenceTransformer('bert-base-nli-mean-tokens')

def cos_similarity(sen_1 : str , sen_2 : str,model=model) -> float:
    
    sentences=[sen_1,sen_2]
    embeddings = model.encode(sentences)
    score = cosine_similarity(embeddings[0].reshape(-1,2), embeddings[1].reshape(-1,2))
    return score[0][0]
