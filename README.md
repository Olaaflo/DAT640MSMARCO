# MS Marco Passage Retrieval - Team - 008 
Project in DAT640 at the University of Stavanger(UIS).  

A full ranking retrieval system is implemented based on first pass
retrieval followed by a re-ranking step. BM25 is used for the first
stage of the retrieval process. Re-ranking is performed using a cross-
encoder neural network called miniLM-L12-v2.  

#### Table of Contents:  
- [Tech/Framework Used](#tech)
- [Short Description](#short-desc)
- [User Guide](#usr-guide)

<a name="tech"></a>
## Tech/Framework Used
- [Elasticsearch](https://elasticsearch-py.readthedocs.io/en/v8.5.0/)
- [Hugging Face - miniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2?text=I+like+you.+I+love+you)

<a name="short-desc"></a>
## Short Description
Script  | Description
------------- | ------------- 
[advanced_method_eval.py]()  | Code for the advanced method.  
[baseline_eval.py]()  | code for baseline evaluation.  
[functions.py]()  | Collection of different functions.  

<a name="usr-guide"></a>
## User Guide:
### 1. Install all packages from requirements.txt  
    `pip install -r requirements.txt`  

### 2. Running the Code 


## Information 
...
