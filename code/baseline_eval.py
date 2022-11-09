#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:36:00 2022

@author: hannahhaland

Perform evaluation using trec eval

"""
from pathlib import Path
from trectools import TrecQrel, TrecRun, TrecEval
from pprint import pprint
from typing import Any, Dict, List, Union, Callable, Set
from elasticsearch import Elasticsearch
import csv
import math
import string
import numpy as np

from functions import get_metrics, get_queries, get_relevance_scores, summarize_metrics

INDEX_NAME = "msmarcopassages"

# Query documents. Contains query id and query text 
QUERIES_TRAIN = str(Path(r"data/queries.train.tsv"))
QUERIES_EVAL = str(Path(r"data/queries.eval.tsv"))
QUERIES_DEV = str(Path(r"data/queries.dev.tsv"))
QUERY_FILES = [QUERIES_DEV, QUERIES_TRAIN, QUERIES_EVAL]

# Evaluation scores
RELEVANCE_SCORES = str(Path(r"data/2019qrels-pass.txt"))

# OUTPUT FILE NAMES (for use by trec_eval)
ADVANCED_METHOD_RESULTS = "advanced_method_results"
BASELINE_RESULTS = "baseline_results"
QRELS_BINARY = "qrels_binary"

def perform_ranking(es: Elasticsearch, index_name: str,  rel_scores, queries, results_file=BASELINE_RESULTS): 
    """
    Iterate through qid-pid-rel tuples (in rel_scores dictionary).
    Perform baseline retrieval on query text.
    Gather metrics for each query. 
    (Use default valules for k in the metric functions.)
    
    Args: 
        es: Elasticsearch object
        index_namme: name of index
        qrel_scores: dictionary containing {qid: {pid: rel}}
        queries: dictionary containing {qid: query_text}
        

    Returns
    -------
        Outputs a csv file what contains the results of the runs in TrecRun format.
        
    TrecRun format

    qid Q0 docno rank score tag
    
    where:
    
    qid is the query number
    Q0 is the literal Q0
    docno is the id of a document returned for qid
    rank (1-999) is the rank of this response for this qid
    score is a system-dependent indication of the quality of the response
    tag is the identifier for the system
    """
    

    result_list = []
    for qid in rel_scores.keys():
        query_text = queries[qid]
        
        # process query text (Elastic search does not function well with escape characters)
        # Replace punctuation marks with single space
        for char in string.punctuation: 
            query_text = query_text.replace(char, " ")
    
        res = es.search(index = index_name, q = query_text, _source = False, size = 1000)["hits"]["hits"]

        result_list.append([[qid, "Q0", hit["_id"], count+1, hit["_score"], "baseline"] for (count, hit) in enumerate(res)])

    # write result to file
    output_file = results_file
    with open(output_file, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile, delimiter = "\t",  quotechar='"',) 
        for i in range(len(result_list)): 
            csvwriter.writerows(result_list[i])
    
    return result_list

def main():
    es = Elasticsearch()    
    es.info()  

    # get queries
    queries = get_queries(QUERY_FILES)
    
    # get relevance scores
    rel_scores = get_relevance_scores(RELEVANCE_SCORES, QRELS_BINARY)

    # baseline retrieval
    result_list = perform_ranking(es, INDEX_NAME, rel_scores, queries, results_file=BASELINE_RESULTS)

    #metrics dictionary
    metrics_dic = get_metrics(results_file=BASELINE_RESULTS)

    # Summarize relevant metrics and print to screen
    summarize_metrics(metrics_dic)

if __name__ == '__main__':
    main()


