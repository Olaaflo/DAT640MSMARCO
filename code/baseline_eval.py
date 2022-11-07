#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:36:00 2022

@author: hannahhaland

Perform evaluation using trec eval

"""
from trectools import TrecQrel, TrecRun, TrecEval
from pprint import pprint
from typing import Any, Dict, List, Union, Callable, Set
from elasticsearch import Elasticsearch
import csv
import math
import string
import numpy as np

INDEX_NAME = "msmarcopassages"

# Query documents. Contains query id and query text 
QUERIES_TRAIN = "../data/queries.train.tsv"
QUERIES_EVAL = "../data/queries.eval.tsv"
QUERIES_DEV = "../data/queries.dev.tsv"
QUERY_FILES = [QUERIES_DEV, QUERIES_TRAIN, QUERIES_EVAL]

# Evaluation scores
RELEVANCE_SCORES = "../data/2019qrels-pass.txt"

# OUTPUT FILE NAMES (for use by trec_eval)
BASELINE_RESULTS = "baseline_results"
QRELS_BINARY = "qrels_binary"

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
    
    A file of the binary relevance scores in correct format for trec_eval is also created.
    data in qrels file need to be in this format: 
        query-id 0 document-id relevance

    """
    rel_scores = {}
    rel_list = []

    with open(RELEVANCE_SCORES) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for count, line in enumerate(tsv_file):
            [qid, x, pid, rel] = line[0].split()
            #put info in dictionary
            try: 
                rel_scores[qid][pid] = math.floor(int(rel)/2)
            except: 
                rel_scores[qid] = {pid:math.floor(int(rel)/2)}
            
            # add line to rel_list
            rel_list.append([qid, 0, pid, rel_scores[qid][pid]])
    
    # create the new qrels file for trec_eval
    output_file = QRELS_BINARY
    with open(output_file, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile, delimiter = "\t",  quotechar='"',) 
        csvwriter.writerows(rel_list)
    
    return rel_scores

        

#%% BASELINE RANKING 

def perform_ranking(es: Elasticsearch, index_name: str,  rel_scores, queries): 
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
    
        res = es.search(index = INDEX_NAME, q = query_text, _source = False, size = 1000)["hits"]["hits"]

        result_list.append([[qid, "Q0", hit["_id"], count+1, hit["_score"], "baseline"] for (count, hit) in enumerate(res)])

    # write result to file
    output_file = BASELINE_RESULTS
    with open(output_file, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile, delimiter = "\t",  quotechar='"',) 
        for i in range(len(result_list)): 
            csvwriter.writerows(result_list[i])
    
    return result_list
    
def get_metrics(results_file = BASELINE_RESULTS, rels_file = QRELS_BINARY):
    """
    use trec eval to compute metrics
    Args: 
        results_file: baseline retrieval results  BASELINE_RESULTS
        rels_file: binary relevance file QRELS_BINARY

    """
    
    # create trecrun object with results file
    res = TrecRun(results_file)
    qrels = TrecQrel(rels_file)
    
    # perform evaluation. this is a pandas df of the metrics
    eval_metrics = res.evaluate_run(qrels, per_query=True).data
    
    # get qids  this includes "all"
    qids = eval_metrics["query"].unique()
    
    # create a dictionary containing results
    metrics_dic = {}
    for q in qids: 
        resq = eval_metrics[eval_metrics["query"] == q].drop(labels=["query"], axis = 1).set_index("metric").to_dict()["value"]
        metrics_dic[q]=resq
   
    return metrics_dic

def summarize_metrics(metrics_dic): 
    """
    Function collects the relevant metrics and print to screen. 
    Relevant metrics are: AP, MDC@10, R@1000, RR
    """
    
    # collect rel-ret and rel results for each query
    num_rel_ret=[]
    num_rel = [metrics_dic[k]["num_rel"] for k in metrics_dic.keys() ]

    for k in metrics_dic.keys(): 
        try: 
            num_rel_ret.append(metrics_dic[k]["num_rel_ret"])
        except: 
            num_rel_ret.append(0)
        
    # Average calculate Recall over all queries
    recall = np.array(num_rel_ret[:-1]) / np.array(num_rel[:-1])
    mean_recall = sum(recall)/len(recall)
    
    # Summarise results and print
    output_metrics = {"AP": round(metrics_dic["all"]["map"], 4),
                      "NDCG@10": round(metrics_dic["all"]["NDCG_10"],4),
                      "R@1000" : round(mean_recall,4),
                      "RR": round(metrics_dic["all"]["recip_rank"],4), 
                      }
    print("Performance of Elasticsearch BM25 retrieval:")
    pprint(output_metrics)
    
    return 

#%% MAIN

es = Elasticsearch()    
es.info()  

 # get queries
queries = get_queries()
 
 # get relevance scores
rel_scores = get_relevance_scores()

# baseline retrieval
result_list = perform_ranking(es, INDEX_NAME, rel_scores, queries)

#metrics dictionary
metrics_dic = get_metrics(BASELINE_RESULTS, QRELS_BINARY)

# Summarize relevant metrics and print to screen
summarize_metrics(metrics_dic)



