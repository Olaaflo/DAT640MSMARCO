#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:53:16 2022

@author: hannahhaland


Code for :
    indexing of MSMarco Passages
    performing baseline retrieval using elasticsearch's BM25 alg
    evaluating metrics (NDCG@10, AP, R@1000, RR)

"""

from typing import Any, Dict, List, Union, Callable, Set
from elasticsearch import Elasticsearch
from pprint import pprint
from retrievals import baseline_retrieval
import csv
import numpy as np
import math
import string

INDEX_NAME = "msmarcopassages"

INDEX_SETTINGS = {
    "mappings": {
    "properties": {
        "body": {
            "type": "text",
            "term_vector": "with_positions",
            "analyzer": "english",
        },
    }
    }
}



# TSV file containing corpus containing 8.8M passages
DATA_FILE = "../data/collection.tsv"

# Query documents. Contains query id and query text apears
QUERIES_TRAIN = "../data/queries.train.tsv"
QUERIES_EVAL = "../data/queries.eval.tsv"
QUERIES_DEV = "../data/queries.dev.tsv"
QUERY_FILES = [QUERIES_DEV, QUERIES_TRAIN, QUERIES_EVAL]

# Evaluation scores
RELEVANCE_SCORES = "../data/2019qrels-pass.txt"

#%% INDEXING

def bulk_index(es: Elasticsearch, data_file=DATA_FILE, 
    index=INDEX_NAME, index_settings=INDEX_SETTINGS, cutoff=np.inf) -> None:
    """
    Iterate over the MSMarco passages dataset and create elasticsearch index.

     Args:
        cutoff: number of items to index before stopping. Indexes all items by 
        default.

    Returns
    -------
    None. (index created)

    """
    
    # if the index exists, delete it
    if es.indices.exists(index=index):    
        es.indices.delete(index=index)
        
    # create the index
    es.indices.create(index=index, body=index_settings)
    
    # create dictionary of passages. keys: doc_id, value: passsage text
    corpus = {}

    i = 0
    with open(data_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for count, line in enumerate(tsv_file):

            if count > cutoff:
                break

            docid, text = line
            #put info in dictionary
            corpus[docid] = {"body": text}
                     
    # Loop through dictionary, pass to es to create index
    for doc_id, text in corpus.items(): 
        es.index(document=text, id=doc_id, index=index)
        

#%% LOAD DATA
def get_queries(): 
    """
    Compile dictionary of query ids and query text from the available query documents.

    Returns
    -------
    Dictiionary of 1010916 queries. keys are qid, values are query strings

    """
    # create dictionary of passages. keys: doc_id, value: passsage text
    queries = {}
    
    for qfile in QUERY_FILES: 
        with open(qfile) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for count, line in enumerate(tsv_file):
                docid, text = line
                #put info in dictionary
                queries[docid] = text
                     
    return queries


def get_relevance_scores(): 
    """
    Preprocess the qid-pid-rel file.
    There are 9259 qid-pid-rel tuples in the 2019 TREC evaluation file.
    Relevance scores are 0,1,2,3. These are converted to 0 (not relevant) and 1 (relevant)

    Returns
    -------
    Dictionary of {qid: {pid: rel}}

    """
    rel_scores = {}

    with open(RELEVANCE_SCORES) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for count, line in enumerate(tsv_file):
            [qid, x, pid, rel] = line[0].split()
            #put info in dictionary
            try: 
                rel_scores[qid][pid] = math.floor(int(rel)/2)
            except: 
                rel_scores[qid] = {pid:math.floor(int(rel)/2)}
                       
    return rel_scores


#%% COMPUTE METRICS

def dcg(relevances: List[int], k: int) -> float:
    """Computes DCG@k, given the corresponding relevance levels for a ranked list of documents.
    
    For example, given a ranking [2, 3, 1] where the relevance levels according to the ground 
    truth are {1:3, 2:4, 3:1}, the input list will be [4, 1, 3].
    
    Args:
        relevances: List with the ground truth relevance levels corresponding to a ranked list of documents.
        k: Rank cut-off.
        
    Returns:
        DCG@k (float).
        
    Ref:  https://colab.research.google.com/drive/1gC2xzubrP3eBkHoCrJU-DQLhAInz8J3k?usp=sharing#scrollTo=P9qGPph637og
    """
    # Note: Rank position is indexed from 1.
    return relevances[0] + sum(
        rel / math.log(i + 2, 2) 
         for i, rel in enumerate(relevances[1:k])
    )

def ndcg(system_ranking: List[int], ground_truth: List[int], k:int = 10) -> float:
    """Computes NDCG@k for a given system ranking.
    
    Args:
        system_ranking: Ranked list of document IDs (from most to least relevant).
        ground_truth: Dict with document ID: relevance level pairs. Document not present here are to be taken with relevance = 0.
        k: Rank cut-off.
    
    Returns:
        NDCG@k (float).
        
    Ref:  https://colab.research.google.com/drive/1gC2xzubrP3eBkHoCrJU-DQLhAInz8J3k?usp=sharing#scrollTo=P9qGPph637og
    """
    # Relevance levels for the ranked docs.
    relevances = [ground_truth.get(rank,0) for rank in system_ranking]

    # Relevance levels of the idealized ranking.
    relevances_ideal = sorted(ground_truth.values(), reverse=True)
    
    return dcg(relevances, k) / dcg(relevances_ideal, k)  

def get_average_precision(
    system_ranking: List[str], ground_truth: Set[str],k : int = 1000) -> float:
    """Computes Average Precision (AP).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.
        k: only consider the first 1000 items returned

    Returns:
        AP (float).
        
    Ref: page 32/69 in Module 5
    """
    if k > len(system_ranking):
        k = len(system_ranking)
    
    # consider only the fist k items returned.
    system_ranking = system_ranking[:k]
    
    # list checking if doc relevant
    correct_retrieved = [1 if doc in ground_truth else 0 for doc in system_ranking ]
    
    # p@k 
    patk = [sum(correct_retrieved[:k])/k for k in range(1,len(system_ranking)+1)]
    
    # only relevant docs contribute to the sum when calculating Average Precision
    filtered_patk = [patk[i] for i in range(len(patk)) if correct_retrieved[i] ]
    
    try:
        AP = sum(filtered_patk) / len(ground_truth)
    except: 
        AP = 0.0
    
    return AP

def get_recall(system_ranking: List[str], ground_truth: Set[str], k: int = 1000
) -> float:
    """
    Get R@k

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.
        k : int, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
        R@k (float)
    """
        
    # list checking if doc relevant
    correct_retrieved = [1 if doc in ground_truth else 0 for doc in system_ranking ]      
    
    # look at first k only
    if k > len(system_ranking): 
        k = len(system_ranking)
    first_k = correct_retrieved[:k]
    
    # Recall at k (#relevant retrieved / # relevant docs)

    Rk = sum(first_k)/len(ground_truth)
    
    return Rk
    

def get_reciprocal_rank(
    system_ranking: List[str], ground_truth: Set[str]
) -> float:
    """Computes Reciprocal Rank (RR).

    Args:
        system_ranking: Ranked list of document IDs.
        ground_truth: Set of relevant document IDs.

    Returns:
        RR (float).
        
    Ref: page38/69 of M5    
    
    """
    
    RR = 0.0 #initialise RR
    
    # RR is the reciprical of the rank at which the first relevant doc is retrieved.
    ## e.g. if first doc is relevant, RR = 1/1 = 1
    ## e.g. if first doc not relevant, second doc is relevant, RR = 1/2
    for i in range(len(system_ranking)): 
        if system_ranking[i] in ground_truth: 
            rank = i+1
            RR = 1/rank
            break   
    
    return RR

#%% PERFORM EVALUATION 

def perform_evaluation(es: Elasticsearch, index_name: str, rel_scores, queries): 
    """
    Iterate through qid-pid-rel tuples (in rel_scores dictionary).
    Perform baseline retrieval on query text.
    Gather metrics for each query. 
    (Use default valules for k in the metric functions.)
    
    Args: 
        es: Elasticsearch object
        index_namme: name of index
        rel_scores: dictionary containing {qid: {pid: rel}}
        queries: dictionary containing {qid: query_text}
        

    Returns
    -------
    [Average RR, Average NDCG@10, Average AP, Average R@1000]

    """
    
    # dictionary to gather the metrics after each iteration
    metrics = {"RR": [], "NDCG10": [], "AP" : [], "R1000": []}
    
    for qid, scores in rel_scores.items(): 
        query_text = queries[qid]
        
        # process query text (Elastic search does not function well with escape characters)
        # Replace punctuation marks with single space
        for char in string.punctuation: 
            query_text = query_text.replace(char, " ")
        
        #get ranked list of first 1000 documents in descending relevance using BM25
        system_ranking = baseline_retrieval(es, INDEX_NAME , query_text, k = 1000)
        
        # get ground truth documents - collect only pids that have positve rel.
        ground_truth = set([k for k,v in scores.items() if v > 0])
    
        # calculate RR, append to metrics
        RR = get_reciprocal_rank(system_ranking, ground_truth)
        metrics["RR"].append(RR)
        
        # calculate NDCG10, append to metrics
        NDCG10 = ndcg(system_ranking, scores)
        metrics["NDCG10"].append(NDCG10)
        
        # calculate average precision
        AP = get_average_precision(system_ranking, ground_truth)
        metrics["AP"].append(AP)
        
        # calculate recall
        R1000 = get_recall(system_ranking, ground_truth)
        metrics["R1000"].append(R1000)
    
    # compute averages for all the metrics
    average_metrics = {"RR": round(sum(metrics["RR"])/len(metrics["RR"]),4),
                       "NDCG@10": round(sum(metrics["NDCG10"])/len(metrics["NDCG10"]),4),
                       "AP": round(sum(metrics["AP"])/len(metrics["AP"]),4),
                       "R@1000": round(sum(metrics["R1000"])/len(metrics["R1000"]),4)
                       }
    
    print("Performance of Elasticsearch BM25 retrieval:")
    pprint(average_metrics)
    
    return 
    
    
#%% MAIN

def main():
    #create Elasticsearch object and perform indexing
    es = Elasticsearch()    
    es.info()  
    
    #SAFE guard this indexing bit once index is created.
# =============================================================================
#     # create index of the corpus of 8.8M passages
#     print("Creating index...")
#     bulk_index(es)
#     print("Index created.")
# =============================================================================
    
    # get queries
    queries = get_queries()
    
    # get relevance scores
    rel_scores = get_relevance_scores()
    
    # perform baseline evaluation
    perform_evaluation(es, INDEX_NAME, rel_scores, queries)
    

if __name__ == "__main__":
    main()