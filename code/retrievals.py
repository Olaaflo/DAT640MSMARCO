from typing import List
from elasticsearch import Elasticsearch
import numpy as np
import torch

#%% BASELINE RETRIEVAL

def baseline_retrieval(
    es: Elasticsearch, index_name: str , query: str, k: int = 1000) -> List[str]:
    """Performs baseline retrival on index.
    
    Function takes a query in the form of a string with space-separated terms, 
    and first building an Elasticsearch query from these, 
    and then retrieving the highest-ranked 
    entities based on that query from the index, and finally returning 
    the names of the top k candidates as a list in descending order according 
    to the score awarded by Elasticsearch's internal BM25 implementation.

    Args:
        es: Elasticsearch instance.
        index_name: A string of text.
        query: A string of text, space separated terms.
        k: An integer.

    Returns:
        A list of entity IDs as strings, up to k of them, in descending order of
            scores.
    
    """

    res = es.search(index = index_name, q = query, _source = False, size = k)
    result_list = [hit["_id"] for hit in res["hits"]["hits"]]
    
    return result_list


def advanced_method(es, index_name: str, baseline: List[str], query: str, model, tokenizer):
    docs = [es.get(index=index_name, id=_id)['_source']['body'] for _id in baseline]

    features = tokenizer([query] * len(baseline), docs,  padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        sorted_indexes = list(reversed(np.argsort(list(scores))))
    
    return [baseline[i] for i in sorted_indexes]