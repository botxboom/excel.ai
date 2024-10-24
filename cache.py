from sentence_transformers import SentenceTransformer, util
import cachetools
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cache = cachetools.LRUCache(maxsize=1000)

SIMILARITY_THRESHOLD = 0.8

def get_embedding(text):
    """
    Get the embedding of the input text using the embedding model.
    """
    return embedding_model.encode(text)

def get_cached_result(query):
    """
    Check the cache for a similar query using cosine similarity.
    """
    query_embedding = get_embedding(query)

    # Iterate through cached items to find the most similar query
    for cached_query, (cached_embedding, cached_result) in cache.items():
        similarity = util.cos_sim(query_embedding, cached_embedding).item()  # Cosine similarity

        if similarity >= SIMILARITY_THRESHOLD:
            return cached_result  # Return cached result if similarity is above threshold

    return None  # Return None if no similar query is found

def cache_result(query, result):
    """
    Cache the query and result along with its embedding.
    """
    query_embedding = get_embedding(query)
    cache[query] = (query_embedding, result)