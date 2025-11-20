from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Any, Callable

def find_top_k_similar_items(
    query: str,
    candidate_list: List[Any],
    embedding_function: Callable,
    top_k: int = 5
) -> List[Any]:

    sims = []
    
    query_vec = embedding_function.embed_query(query)
    
    item_texts = [str(item) for item in candidate_list]
    item_vectors = embedding_function.embed_documents(item_texts) 
    
    for item, item_vec in zip(candidate_list, item_vectors):
        sim = cosine_similarity([item_vec], [query_vec])[0][0]
        sims.append((item, sim))
    
    sims.sort(key = lambda x:[1], reverse = True)
    top_k_list = [item for item, sim in sims[:top_k]]
    
    return top_k_list