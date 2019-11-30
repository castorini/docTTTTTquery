# docTTTTTquery

docTTTTTquery is the latest version of doc2query family of document expansion models.
The basic idea is to train a model, that when given an input document, generates questions that the document might answer (or more broadly, queries for which the document might be relevant).
These predicted questions (or queries) are then appended to the original documents, which are then indexed as before.
docTTTTTquery gets its name from the use of T5 as the expansion model.

The primary advantage of this approach is that expensive neural inference is pushed to _indexing time_, which means that "bag of words" queries against an inverted index built on the augmented document collection are only slightly slower (due to longer documents) &mdash; but the retrieval results are _much_ better.
Of course, these documents can be further reranked by another neural model in a multi-stage ranking architecture.

The results on the MS MARCO show that docTTTTTquery is as effective as the best non-BERT ranking model while increasing latency only slightly compared to vanila BM25:

MSMARCO Passage Ranking Leaderboard (Nov 30th 2019) | Eval MRR@10 | Latency
------------------------------------- | :------: | :------:
[BM25 + BERT](https://github.com/nyu-dl/dl4marco-bert) | 36.8 | 3500 ms
docTTTTTquery (this code)             | 27.2 | 64 ms
[best non-BERT](https://github.com/sebastian-hofstaetter/sigir19-neural-ir) | 27.7 | -
[doc2query](https://github.com/nyu-dl/dl4ir-doc2query)              | 21.8 | 61 ms
[BM25](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)  | 18.6  | 55 ms

## Training T5

The following command will train a T5-base model for 4k iterations to predict queries from documents. We assume you put the tsv training file in `gs://your_bucket/data/doc_query_pairs.train.tsv`. Also, please change `your_tpu_name`, `your_project_id`, and `your_bucket` accordingly.

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

## Download Predicted Queries

[Download the predicted queries here](https://storage.cloud.google.com/doctttttquery_git/predicted_queries_topk_sampling.zip?cloudshell=true). This file contains 80 sampled queries draw with the top-k sampling decoder.

## Expanding documents
Before appending the sampled queries to the documents, we need to concatenate them into a file that will contain all the samples for the same document in a single line:
```
for i in {000..017}; do
    echo "Processing chunk $i"
    paste -d" " predicted_queries_topk_sample???.txt${i}-1004000 \
    > predicted_queries_topk.txt${i}-1004000
done

cat predicted_queries_topk.txt???-1004000 > predicted_queries_topk.txt-1004000
```

We can now append those queries to the original documents:
```
TODO
```
