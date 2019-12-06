# docTTTTTquery

[Link to the paper](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery.pdf)

docTTTTTquery is the latest version of doc2query family of document expansion models.
The basic idea is to train a model, that when given an input document, generates questions that the document might answer (or more broadly, queries for which the document might be relevant).
These predicted questions (or queries) are then appended to the original documents, which are then indexed as before.
docTTTTTquery gets its name from the use of T5 as the expansion model.

The primary advantage of this approach is that expensive neural inference is pushed to _indexing time_, which means that "bag of words" queries against an inverted index built on the augmented document collection are only slightly slower (due to longer documents) &mdash; but the retrieval results are _much_ better.
Of course, these documents can be further reranked by another neural model in a multi-stage ranking architecture.

The results on the MS MARCO show that docTTTTTquery is way more effective than doc2query and as effective as the best non-BERT ranking model while increasing latency (time to retrieve 1000 docs per query) only slightly compared to vanilla BM25:

MS MARCO Passage Ranking Leaderboard (Nov 30th 2019) | Eval MRR@10 | Latency
------------------------------------- | :------: | ------:
[BM25 + BERT](https://github.com/nyu-dl/dl4marco-bert) | 36.8 | 3500 ms
[best non-BERT](https://github.com/sebastian-hofstaetter/sigir19-neural-ir) | 27.7 | -
docTTTTTquery (this code)             | 27.2 | 64 ms
[doc2query](https://github.com/nyu-dl/dl4ir-doc2query)              | 21.8 | 61 ms
[BM25](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)  | 18.6  | 55 ms

## Data and Trained Models

We make the following data available for download:

+ `doc_query_pairs.train.tsv`: Approximately 500,000 pairs of passage-query pairs used to train the model.
+ `queries.dev.small.tsv`: 6,980 queries from the MS MARCO dev set. In this tsv file, the first column is the query id, and the second is the query text.
+ `qrels.dev.small.tsv`: 7,437 pairs of query relevant passage ids from the MS MARCO dev set. In this tsv file, the first column is the query id, and the third column is the passage id. The other two columns (second and fourth) are not used.
+ `collection.tar.gz`: All passages (8,841,823) in the MS MARCO corpus. In this tsv file, the first column is the passage id, and the second is the passage text.
+ `predicted_queries_topk_sampling.zip`: 80 predicted queries for each MS MARCO passage, using T5-base and top-_k_ sampling.
+ `run.dev.small.tsv`:  Approximately 6,980,000 pairs of dev set queries and retrieved passages using the passages expanded with docTTTTTquery + BM25. In this tsv file, the first column is the query id, the second column is the passage id, and the third column is the rank of the passage. There are 1000 passages per query in this file.
+ `t5-base.zip`: trained T5 model used for generating the expansions.
+ `t5-large.zip`: larger trained T5 model; we didn't find the output to be any better.

Download and verify the above files from the below table:

File | Size | MD5 | Download
:----|-----:|:----|:-----
`doc_query_pairs.train.tsv` | 197 MB | `aa673014f93d43837ca4525b9a33422c` | [[GCS](https://storage.googleapis.com/doctttttquery_git/doc_query_pairs.train.tsv)] [[Dropbox](https://www.dropbox.com/s/5i64irveqvvegey/doc_query_pairs.train.tsv)]
`queries.dev.small.tsv` | 283 KB | `41e980d881317a4a323129d482e9f5e5` | [[GCS](https://storage.googleapis.com/doctttttquery_git/queries.dev.small.tsv)] [[Dropbox](https://www.dropbox.com/s/hq6xjhswiz60siu/queries.dev.small.tsv)]
`qrels.dev.small.tsv` | 140 KB| `38a80559a561707ac2ec0f150ecd1e8a` | [[GCS](https://storage.googleapis.com/doctttttquery_git/qrels.dev.small.tsv)] [[Dropbox](https://www.dropbox.com/s/khsplt2fhqwjs0v/qrels.dev.small.tsv)]
`collection.tar.gz` | 987 MB | `87dd01826da3e2ad45447ba5af577628` | [[GCS](https://storage.googleapis.com/doctttttquery_git/collection.tar.gz)] [[Dropbox](https://www.dropbox.com/s/lvvpsx0cjk4vemv/collection.tar.gz)]
`predicted_queries_topk_sampling.zip` | 7.9 GB | `8bb33ac317e76385d5047322db9b9c34` | [[GCS](https://storage.cloud.google.com/doctttttquery_git/predicted_queries_topk_sampling.zip)] [[Dropbox](https://www.dropbox.com/s/uzkvv4gpj3a596a/predicted_queries_topk_sampling.zip)]
`run.dev.small.tsv` | 133 MB | `d6c09a6606a5ed9f1a300c258e1930b2` | [[GCS](https://storage.cloud.google.com/doctttttquery_git/run.dev.small.tsv)] [[Dropbox](https://www.dropbox.com/s/nc1drdkjpxxsngg/run.dev.small.tsv)]
`t5-base.zip` | 357 MB | `881d3ca87c307b3eac05fae855c79014` | [[GCS](https://storage.googleapis.com/doctttttquery_git/t5-base.zip)] [[Dropbox](https://www.dropbox.com/s/q1nye6wfsvf5sen/t5-base.zip)]
`t5-large.zip` | 1.2 GB | `21c7e625210b0ae872679bc36ed92d44` | [[GCS](https://storage.googleapis.com/doctttttquery_git/t5-large.zip)] [[Dropbox](https://www.dropbox.com/s/gzq8r68uk38bmum/t5-large.zip)]

## Installation

Note: if you plan to train or infer with T5, keep in mind that it only works on TPUs (and consequently Google Cloud machines), so this installation must be performed on a Google Cloud instance. If you only want to reproduce our results, you only need to install the search engine framework (Anserini), described below.

You first need to install t5 (please check the [original T5 repository](https://github.com/google-research/text-to-text-transfer-transformer) for updated installation instructions):
```
pip install t5[gcp]
```

You also need to install [Anserini](https://github.com/castorini/anserini), a search engine framework that will index and retrieve passages:
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

Download `queries.dev.small.tsv`, `qrels.dev.small.tsv`, `collection.tar.gz`, and `predicted_queries_topk_sampling.zip` using one of the options above.

Before appending the sampled queries to the passages, we need to concatenate them.
The commands below create a file that contains 40 concatenated samples per line and 8,841,823 lines, one for each passage in the corpus.
We concatenate only the first 40 samples as there is only a tiny gain in MRR@10 when using 80 samples (nevertheless, we provide 80 samples in case researchers want to use this data for other purposes).

```bash
unzip predicted_queries_topk_sampling.zip

for i in $(seq -f "%03g" 0 17); do
    echo "Processing chunk $i"
    paste -d" " predicted_queries_topk_sample0[0-3]?.txt${i}-1004000 \
    > predicted_queries_topk.txt${i}-1004000
done

cat predicted_queries_topk.txt???-1004000 > predicted_queries_topk.txt-1004000
```

We can now append those queries to the original MS MARCO passage collection:

```bash
tar -xvf collection.tar.gz

python convert_collection_to_jsonl.py \
    --collection_path=collection.tsv \
    --predictions=predicted_queries_topk.txt-1004000 \
    --output_folder=./docs
```

We will now create an index in Anserini for the 8,841,823 expanded docs (replace `/path/to/anserini/` with actual location of Anserini):

```bash
sh /path/to/anserini/target/appassembler/bin/IndexCollection \
  -collection JsonCollection -generator LuceneDocumentGenerator \
  -threads 9 -input ./docs -index ./lucene-index
```

Once the expanded passages are indexed, we can retrieve 1000 passages per query in MS MARCO dev set:

```
python -u $HOME/anserini/src/main/python/msmarco/retrieve.py \
  --index ./lucene-index \
  --qid_queries ./queries.dev.small.tsv \
  --output ./run.dev.small.tsv \
  --hits 1000
```

We evaluate the results using the MS MARCO eval script:
```
python anserini/src/main/python/msmarco/msmarco_eval.py \
  ./qrels.dev.small.tsv \
  ./run.dev.small.tsv
```

The output should be similar to:
```
#####################
MRR @10: 0.2767497271114737
QueriesRanked: 6980
#####################
```

## Training T5

The following command will train a T5-base model for 4k iterations to predict queries from passages. We assume you put the tsv training file in `gs://your_bucket/data/doc_query_pairs.train.tsv`. Also, please change `your_tpu_name`, `your_tpu_zone`, `your_project_id`, and `your_bucket` accordingly.

```
t5_mesh_transformer  \
  --tpu="your_tpu_name" \
  --gcp_project="your_project_id" \
  --tpu_zone="your_tpu_zone" \
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

## Predicting Queries from Passages
We first need to prepare an input file that contains one passage text per line. We achieve this by extracting the second column of `collection.tsv`:
```
cut -f1 collection.tsv > input_docs.txt
```
We also need to split the file into smaller files (each with 1M lines) to avoid TensorFlow complaining that proto arrays can only have 2GB:
```
split --suffix-length 2 --numeric-suffixes --lines 1000000 input_docs.txt input_docs.txt
```

We now upload the files to Google Cloud Storage:
```
gsutil cp input_docs.txt?? gs://your_bucket/data/
```

We are now ready to predict queries from passages. Remember to replace `your_tpu`, `your_tpu_zone`, `your_project_id` and `your_bucket` with your values. Note that the command below will only sample one query per passage. If you want multiple samples, you will need to repeat this process multiple times (remember to replace `decode_from_file.output_filename` with a new filename for each sample).
```
for ITER in {00..09}; do
    t5_mesh_transformer \
      --tpu="your_tpu" \
      --gcp_project="your_project_id" \
      --tpu_zone="your_tpu_zone" \
      --model_dir="gs://your_bucket/models/" \
      --gin_file="gs://neuralresearcher_data/t5-data/pretrained_models/base/operative_config.gin" \
      --gin_file="infer.gin" \
      --gin_file="sample_decode.gin" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
      --gin_param="infer_checkpoint_step = 1004000" \
      --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 64}" \
      --gin_param="Bitransformer.decode.max_decode_length = 64" \
      --gin_param="decode_from_file.input_filename = 'gs://your_bucket/data/input_docs.txt$ITER'" \
      --gin_param="decode_from_file.output_filename = 'gs://your_bucket/data/predicted_queries_topk_sample.txt$ITER'" \
      --gin_param="tokens_per_batch = 131072" \
      --gin_param="Bitransformer.decode.temperature = 1.0" \
      --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = 10"
done
```


