import csv
import string
from pathlib import Path
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from elasticsearch import Elasticsearch
from functions import (
        baseline_retrieval, advanced_method, bulk_index, 
        get_queries, get_relevance_scores, summarize_metrics, get_metrics
)

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
QRELS_BINARY = "qrels_binary"



def perform_ranking(es: Elasticsearch, index_name: str,  rel_scores, queries, k=1000, results_file=ADVANCED_METHOD_RESULTS): 
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

        model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        baseline_ranking = baseline_retrieval(es, index_name, query=query_text, k=k)
        
        # Reranking
        if len(baseline_ranking) > 0:
            res = advanced_method(es, index_name, query=query_text, baseline=baseline_ranking, 
                                model=model, tokenizer=tokenizer)
            result_list.append(
                [[qid, "Q0", hit["_id"], count+1, hit["_score"], "baseline"] 
                for (count, hit) in enumerate(res)]
            )

    # write result to file
    output_file = results_file
    with open(output_file, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile, delimiter = "\t",  quotechar='"',) 
        for i in range(len(result_list)): 
            csvwriter.writerows(result_list[i])
    
    return result_list
    

def main():
    # es = Elasticsearch(request_timeout=30, max_retries=10, retry_on_timeout=True)    
    es = Elasticsearch()    
    print(es.info())

    print('Indexing ...')
    # TODO: change rieindex_if_exist to 
    # False when indexing is completed the first time
    bulk_index(es, index=INDEX_NAME, reindex_if_exist=False, batch_size=100_000)
    print('Finished indexing')

    # get queries
    print('Fetching queries ...')
    queries = get_queries(QUERY_FILES)
    print('Finished fetching queries')
    
    # get relevance scores
    print('Fetching rel-scores ...')
    rel_scores = get_relevance_scores(RELEVANCE_SCORES)
    print('Finished fetching rel-scores')

    # baseline retrieval
    print('Performing ranking ...')
    result_list = perform_ranking(es, INDEX_NAME, rel_scores, queries)
    print('Finished getting rankings')

    #metrics dictionary
    metrics_dic = get_metrics(ADVANCED_METHOD_RESULTS, QRELS_BINARY)

    # Summarize relevant metrics and print to screen
    summarize_metrics(metrics_dic)


if __name__ == '__main__':
    main()



