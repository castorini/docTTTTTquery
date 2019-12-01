# docTTTTTquery

docTTTTTquery is the latest version of doc2query family of document expansion models.
The basic idea is to train a model, that when given an input document, generates questions that the document might answer (or more broadly, queries for which the document might be relevant).
These predicted questions (or queries) are then appended to the original documents, which are then indexed as before.
docTTTTTquery gets its name from the use of T5 as the expansion model.

The primary advantage of this approach is that expensive neural inference is pushed to _indexing time_, which means that "bag of words" queries against an inverted index built on the augmented document collection are only slightly slower (due to longer documents) &mdash; but the retrieval results are _much_ better.
Of course, these documents can be further reranked by another neural model in a multi-stage ranking architecture.

The results on the MS MARCO show that docTTTTTquery is as effective as the best non-BERT ranking model while increasing latency (time to retrieve 1000 docs per query) only slightly compared to vanila BM25:

MS MARCO Passage Ranking Leaderboard (Nov 30th 2019) | Eval MRR@10 | Latency
------------------------------------- | :------: | ------:
[BM25 + BERT](https://github.com/nyu-dl/dl4marco-bert) | 36.8 | 3500 ms
[best non-BERT](https://github.com/sebastian-hofstaetter/sigir19-neural-ir) | 27.7 | -
docTTTTTquery (this code)             | 27.2 | 64 ms
[doc2query](https://github.com/nyu-dl/dl4ir-doc2query)              | 21.8 | 61 ms
[BM25](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)  | 18.6  | 55 ms

## Data and Trained Models

We make the following data available for download:

+ `doc_query_pairs.train.tsv`: Approximately 500,000 pairs of document-query pairs used to train the model.
+ `msmarco_docs.txt.zip`: All documents (8.8M) in the MS MARCO corpus.
+ `predicted_queries_topk_sampling.zip`: 80 predicted queries for each MS MARCO passage, using T5-base and top-_k_ sampling.
+ `t5-base.zip`: trained T5 model used for generating the expansions.
+ `t5-large.zip`: larger trained T5 model; we didn't find the output to be any better.

Download and verify the above files from the below table:

File | Size | MD5 | Download
:----|-----:|:----|:-----
`doc_query_pairs.train.tsv` | 197 MB | `aa673014f93d43837ca4525b9a33422c` | [[GCS](https://storage.googleapis.com/doctttttquery_git/doc_query_pairs.train.tsv)]
`msmarco_docs.txt.zip` | 961 MB | `dff69f506ffb1ea84f3e39cd071ee251` | [[GCS](https://storage.googleapis.com/doctttttquery_git/msmarco_docs.txt.zip)] 
`predicted_queries_topk_sampling.zip` | 7.9 GB | `8bb33ac317e76385d5047322db9b9c34` | [[GCS](https://storage.cloud.google.com/doctttttquery_git/predicted_queries_topk_sampling.zip)] [[Dropbox](https://www.dropbox.com/s/uzkvv4gpj3a596a/predicted_queries_topk_sampling.zip)]
`t5-base.zip` | 357 MB | `881d3ca87c307b3eac05fae855c79014` | [[GCS](https://storage.googleapis.com/doctttttquery_git/t5-base.zip)] [[Dropbox](https://www.dropbox.com/s/q1nye6wfsvf5sen/t5-base.zip)]
`t5-large.zip` | 1.2 GB | `21c7e625210b0ae872679bc36ed92d44` | [[GCS](https://storage.googleapis.com/doctttttquery_git/t5-large.zip)] [[Dropbox](https://www.dropbox.com/s/gzq8r68uk38bmum/t5-large.zip)]

## Installation

T5 only works on TPUs (and consequently Google Cloud machines) so this installation must be performed on a Google Cloud instance.
You first need to install t5 (please check the [original T5 repository](https://github.com/google-research/text-to-text-transfer-transformer) for updated installation instructions):
```
pip install t5[gcp]
```

You also need to install [Anserini](https://github.com/castorini/anserini), a search engine framework that will index and retrieve documents:
```
sudo apt-get install maven
git clone https://github.com/castorini/Anserini.git
cd Anserini
mvn clean package appassembler:assemble
tar xvfz eval/trec_eval.9.0.4.tar.gz -C eval/ && cd eval/trec_eval.9.0.4 && make
cd ../ndeval && make
```

## Replicating docTTTTTquery with Anserini

First, we provide instructions on how to replicate our docTTTTTquery runs with Anserini.

Download `predicted_queries_topk_sampling.zip` using one of the options above.
This file contains 80 sampled queries draw with the top-_k_ sampling decoder.

Before appending the sampled queries to the documents, we need to concatenate them into a file that will contain all the samples for the same document in a single line. We will concatenate only the first 40 samples as there no gain when using 80 samples (we, however, provide 80 samples in case researchers want to use this data for other purposes):
```
for i in {000..017}; do
    echo "Processing chunk $i"
    paste -d" " predicted_queries_topk_sample0[0-3]?.txt${i}-1004000 \
    > predicted_queries_topk.txt${i}-1004000
done

cat predicted_queries_topk.txt???-1004000 > predicted_queries_topk.txt-1004000
```

We can now append those queries to the original documents:
```
TODO
```

## Training T5

The following command will train a T5-base model for 4k iterations to predict queries from documents. We assume you put the tsv training file in `gs://your_bucket/data/doc_query_pairs.train.tsv`. Also, please change `your_tpu_name`, `your_project_id`, and `your_bucket` accordingly. . 

```
t5_mesh_transformer  \
  --tpu="your_tpu_name" \
  --gcp_project="your_project_id" \
  --tpu_zone="us-central1-b" \
  --model_dir="gs://your_bucket/models/" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/base/model.ckpt-999900'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 1" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://your_bucket/data/doc_query_pairs.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1004000" \
  --gin_param="tokens_per_batch = 131072"
```

## Inference
TODO


