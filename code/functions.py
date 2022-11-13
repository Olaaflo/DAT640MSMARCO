import csv
import math
import numpy as np
import torch
from typing import List
from elasticsearch import Elasticsearch
from pprint import pprint
from trectools import TrecQrel, TrecRun, TrecEval
from pathlib import Path

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
DATA_FILE = str(Path("../data/collection.tsv"))

# Query documents. Contains query id and query text 
QUERIES_TRAIN = str(Path(r"../data/queries.train.tsv"))
QUERIES_EVAL = str(Path(r"../data/queries.eval.tsv"))
QUERIES_DEV = str(Path(r"../data/queries.dev.tsv"))
QUERY_FILES = [QUERIES_DEV, QUERIES_TRAIN, QUERIES_EVAL]

# Evaluation scores
RELEVANCE_SCORES = str(Path(r"../data/2019qrels-pass.txt"))

# OUTPUT FILE NAMES (for use by trec_eval)
ADVANCED_METHOD_RESULTS = "advanced_method_results"
QRELS_BINARY = "qrels_binary"

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

    res = es.search(index=index_name, q=query, _source=False, size=k)
    
    result_list = [hit["_id"] for hit in res["hits"]["hits"]]
    return result_list

def re_ranker(es, index_name: str, baseline: List[str], query: str, model, tokenizer):
    """Performs re-ranking of the baseline results

    Function takes a list of ranked passages, and reranks them with a 
    re-ranking model after tokinizing the text with the tokenizer given
    as input.

    Args:
        es: Elasticsearch instance.
        index_name: A string of text.
        baseline: List of passage-ids already ranked
        query: A string of text, space separated terms.
        model: A pretrained model used for re-ranking, uses query and passage 
            as input
        tokenizer: A pretrained tokenizer used on input to the model
        
    """
    docs = [es.get(index=index_name, id=_id)["_source"]["body"] for _id in baseline]

    features = tokenizer([query] * len(baseline), docs,  padding=True, truncation=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        scores = model(**features).logits
        scores = [t.item() for t in scores]
        sorted_indexes = list(reversed(np.argsort(list(scores))))

    return [{"_id" : baseline[i], "_score" : scores[i]} for i in sorted_indexes]


def bulk_index(
    es: Elasticsearch, data_file=DATA_FILE, index=INDEX_NAME, 
    batch_size=100_000, index_settings=INDEX_SETTINGS, 
    cutoff=np.inf, reindex_if_exist=True) -> None:
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
        if reindex_if_exist:    
            es.indices.delete(index=index)
        else:
            print("""
            index already exists, aborting...
            Change reindex_if_exist to False 
            if you still want to delete the current index, 
            and reindex the whole thing""")
            return
        
    # create the index
    es.indices.create(index=index, body=index_settings)
    
    # create dictionary of passages. keys: doc_id, value: passsage text
    num_indexed = 0
    batch = 0
    bulk_data = []


    with open(data_file) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for count, line in enumerate(tsv_file):            
            if count > cutoff:
                break
            
            docid, text = line
            bulk_data.append({"index": {"_index": index, "_id": docid}})
            bulk_data.append({"body": text})
            batch += 1

            if batch >= batch_size:
                es.bulk(index=index, body=bulk_data, refresh=True)
                num_indexed += batch
                batch = 0
                bulk_data = []
                print(f"Indexed {num_indexed} passages")

        # Bulk remaining (if remaining)
        if len(bulk_data) > 0:
            es.bulk(index=index, body=bulk_data, refresh=True)
            print(f"Indexed {num_indexed + batch} passages")

def one_by_one_index(es: Elasticsearch, data_file=DATA_FILE, 
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

def get_metrics(results_file=ADVANCED_METHOD_RESULTS, rels_file = QRELS_BINARY):
    """
    use trec eval to compute metrics
    Args: 
        results_file: baseline retrieval results  ADVANCED_METHOD_RESULTS
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


def get_queries(query_files=QUERY_FILES): 
    """
    Compile dictionary of query ids and query text from the available query documents.

    Returns
    -------
    Dictiionary of 1010916 queries. keys are qid, values are query strings

    """
    # create dictionary of passages. keys: doc_id, value: passsage text
    queries = {}
    
    for qfile in query_files: 
        with open(qfile) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            for count, line in enumerate(tsv_file):
                docid, text = line
                #put info in dictionary
                queries[docid] = text
                     
    return queries

def get_relevance_scores(
    relevance_scores=RELEVANCE_SCORES, qrels_binary=QRELS_BINARY): 
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

    with open(relevance_scores) as file:
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
    output_file = qrels_binary
    with open(output_file, "w") as csvfile: 
        csvwriter = csv.writer(csvfile, delimiter = "\t",  quotechar='"',) 
        csvwriter.writerows(rel_list)
    
    return rel_scores

def get_metrics(results_file=ADVANCED_METHOD_RESULTS, rels_file = QRELS_BINARY):
    """
    use trec eval to compute metrics
    Args: 
        results_file: baseline retrieval results  ADVANCED_METHOD_RESULTS
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