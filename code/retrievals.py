from typing import List
from elasticsearch import Elasticsearch

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

def advanced_method():
    assert NotImplementedError("a function for the advanced re-ranking method is not implemented yet")
