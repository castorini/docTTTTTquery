# Document Expansion by Query Prediction

The repo describes experiments with docTTTTTquery (sometimes written as docT5query or doc2query-T5), the latest version of the doc2query family of document expansion models.
The basic idea is to train a model, that when given an input document, generates questions that the document might answer (or more broadly, queries for which the document might be relevant).
These predicted questions (or queries) are then appended to the original documents, which are then indexed as before.
docTTTTTquery gets its name from the use of T5 as the expansion model.

The primary advantage of this approach is that expensive neural inference is pushed to _indexing time_, which means that "bag of words" queries against an inverted index built on the augmented document collection are only slightly slower (due to longer documents) &mdash; but the retrieval results are _much_ better.
Of course, these documents can be further reranked by another neural model in a [multi-stage ranking architecture](https://arxiv.org/abs/1910.14424).

The results on the [MS MARCO passage retrieval task](http://www.msmarco.org/leaders.aspx) show that docTTTTTquery is much more effective than doc2query and (almost) as effective as the best non-BERT ranking model, while increasing query latency (time to retrieve 1000 docs per query) only slightly compared to vanilla BM25:

MS MARCO Passage Ranking Leaderboard (Nov 30th 2019) | Eval MRR@10 | Latency
:------------------------------------ | :------: | ------:
[BM25 + BERT](https://github.com/nyu-dl/dl4marco-bert) from [(Nogueira et al., 2019)](https://arxiv.org/abs/1904.08375) | 36.8 | 3500 ms
FastText + Conv-KNRM (Single) [(Hofstätter et al. SIGIR 2019)](https://github.com/sebastian-hofstaetter/sigir19-neural-ir) (best non-BERT) | 27.7 | -
docTTTTTquery (this code)             | 27.2 | 64 ms
DeepCT [(Dai and Callan, 2019)](https://github.com/AdeDZY/DeepCT)              | 23.9 | 55 ms
doc2query [(Nogueira et al., 2019)](https://github.com/nyu-dl/dl4ir-doc2query)              | 21.8 | 61 ms
[BM25](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)  | 18.6  | 55 ms

For more details, check out our paper:

+ Rodrigo Nogueira and Jimmy Lin.  [From doc2query to docTTTTTquery.](https://cs.uwaterloo.ca/~jimmylin/publications/Nogueira_Lin_2019_docTTTTTquery-v2.pdf)

Why's the paper so short? Check out [our proposal for micropublications](https://github.com/lintool/guide/blob/master/micropublications.md)!

## Quick Links

+ [Data and Trained Models: MS MARCO Passage Dataset](#Data-and-Trained-Models-MS-MARCO-Passage-Dataset)
+ [Replicating MS MARCO Passage Retrieval Results with Anserini](#Replicating-MS-MARCO-Passage-Retrieval-Results-with-Anserini)
+ [Predicting Queries from Passages: T5 Inference with PyTorch](#Predicting-Queries-from-Passages-T5-Inference-with-PyTorch)
+ [Predicting Queries from Passages: T5 Inference with TensorFlow](#Predicting-Queries-from-Passages-T5-Inference-with-TensorFlow)
+ [Learning a New Prediction Model: T5 Training with TensorFlow](#Learning-a-New-Prediction-Model-T5-Training-with-TensorFlow)
+ [Replicating MS MARCO Document Retrieval Results with Anserini](#Replicating-MS-MARCO-Document-Retrieval-Results-with-Anserini)
+ [Predicting Queries from Documents: T5 Inference with TensorFlow](#Predicting-Queries-from-Documents-T5-Inference-with-TensorFlow)

## Data and Trained Models: MS MARCO Passage Dataset

The basic docTTTTTquery model is trained on the MS MARCO passage dataset.
We make the following data and models available for download:

+ `doc_query_pairs.train.tsv`: Approximately 500,000 passage-query pairs used to train the model.
+ `queries.dev.small.tsv`: 6,980 queries from the MS MARCO dev set. In this tsv file, the first column is the query id and the second is the query text.
+ `qrels.dev.small.tsv`: 7,437 pairs of query relevant passage ids from the MS MARCO dev set. In this tsv file, the first column is the query id and the third column is the passage id. The other two columns (second and fourth) are not used.
+ `collection.tar.gz`: All passages (8,841,823) in the MS MARCO passage corpus. In this tsv file, the first column is the passage id and the second is the passage text.
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
`predicted_queries_topk_sampling.zip` | 7.9 GB | `8bb33ac317e76385d5047322db9b9c34` | [[GCS](https://storage.googleapis.com/doctttttquery_git/predicted_queries_topk_sampling.zip)] [[Dropbox](https://www.dropbox.com/s/uzkvv4gpj3a596a/predicted_queries_topk_sampling.zip)]
`run.dev.small.tsv` | 133 MB | `d6c09a6606a5ed9f1a300c258e1930b2` | [[GCS](https://storage.googleapis.com/doctttttquery_git/run.dev.small.tsv)] [[Dropbox](https://www.dropbox.com/s/nc1drdkjpxxsngg/run.dev.small.tsv)]
`t5-base.zip` | 357 MB | `881d3ca87c307b3eac05fae855c79014` | [[GCS](https://storage.googleapis.com/doctttttquery_git/t5-base.zip)] [[Dropbox](https://www.dropbox.com/s/q1nye6wfsvf5sen/t5-base.zip)]
`t5-large.zip` | 1.2 GB | `21c7e625210b0ae872679bc36ed92d44` | [[GCS](https://storage.googleapis.com/doctttttquery_git/t5-large.zip)] [[Dropbox](https://www.dropbox.com/s/gzq8r68uk38bmum/t5-large.zip)]

## Replicating MS MARCO Passage Retrieval Results with Anserini

We provide instructions on how to replicate our docTTTTTquery runs with the [Anserini](https://github.com/castorini/anserini) IR toolkit, using the predicted queries provided above.

First, install Anserini (see [homepage](https://github.com/castorini/anserini) for more details):

```bash
sudo apt-get install maven
git clone https://github.com/castorini/Anserini.git
cd Anserini
mvn clean package appassembler:assemble
tar xvfz eval/trec_eval.9.0.4.tar.gz -C eval/ && cd eval/trec_eval.9.0.4 && make
cd ../ndeval && make
```

Next, download `queries.dev.small.tsv`, `qrels.dev.small.tsv`, `collection.tar.gz`, and `predicted_queries_topk_sampling.zip` using one of the options above.

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

python convert_msmarco_passage_to_anserini.py \
    --collection_path=collection.tsv \
    --predictions=predicted_queries_topk.txt-1004000 \
    --output_folder=./docs
```

Now, create an index using Anserini on the expanded passages (replace `/path/to/anserini/` with actual location of Anserini):

```bash
sh /path/to/anserini/target/appassembler/bin/IndexCollection \
  -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
  -threads 9 -input ./docs -index ./lucene-index
```

Once the expanded passages are indexed, we can retrieve 1000 passages per query for the MS MARCO dev set:

```bash
sh /path/to/anserini/target/appassembler/bin/SearchMsmarco \
  -index ./lucene-index -qid_queries ./queries.dev.small.tsv \
  -output ./run.dev.small.tsv -hits 1000
```

Finally, we evaluate the results using the MS MARCO eval script:

```bash
python /path/to/anserini/src/main/python/msmarco/msmarco_eval.py ./qrels.dev.small.tsv ./run.dev.small.tsv
```

The results should be:

```
#####################
MRR @10: 0.2767497271114737
QueriesRanked: 6980
#####################
```

Voilà!

## Predicting Queries from Passages: T5 Inference with PyTorch

We will use the excelent 🤗 transformers library to sampe queries from our T5 model.

First, install the library:
```bash
pip install transformers
```

Download and unzip `t5-base.zip` from the table above, and load the model checkpoint:
```python
import torch
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(
    'model.ckpt-1004000', from_tf=True, config=config)
model.to(device)
```

Sample 3 questions from a example document (don't forget to append the end-of-sequence token `</s>`):
```python
doc_text = 'The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated. </s>'

input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
outputs = model.generate(
    input_ids=input_ids,
    max_length=64,
    do_sample=True,
    top_k=10,
    num_return_sequences=3)

for i in range(3):
    print(f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')
```

The output should be similar to this:
```
sample 1: why was the manhattan project successful
sample 2: the manhattan project what it means
sample 3: what was the most important aspect of the manhattan project
```

For more information on how to use T5 with HuggingFace's transformers library, [check their documentation](https://huggingface.co/transformers/model_doc/t5.html).

## Predicting Queries from Passages: T5 Inference with TensorFlow

Next, we provide instructions on how to use our trained T5 models to predict queries for each of the 8.8M documents in the MS MARCO corpus.
To speed up inference, we will use TPUs (and consequently Google Cloud machines), so this installation must be performed on a Google Cloud instance.

To begin, install T5 (check the [original T5 repository](https://github.com/google-research/text-to-text-transfer-transformer) for the latest installation instructions):

```bash
pip install t5[gcp]
```

We first need to prepare an input file that contains one passage text per line. We achieve this by extracting the second column of `collection.tsv`:

```bash
cut -f1 collection.tsv > input_docs.txt
```

We also need to split the file into smaller files (each with 1M lines) to avoid TensorFlow complaining that proto arrays can only be 2GB at the most:

```bash
split --suffix-length 2 --numeric-suffixes --lines 1000000 input_docs.txt input_docs.txt
```

We now upload the input docs to Google Cloud Storage:

```bash
gsutil cp input_docs.txt?? gs://your_bucket/data/
```

We also need to upload our trained t5-base model to GCS:

```bash
wget https://storage.googleapis.com/doctttttquery_git/t5-base.zip
unzip t5-base.zip
gsutil cp model.ckpt-1004000* gs://your_bucket/models/
```

We are now ready to predict queries from passages. Remember to replace `your_tpu`, `your_tpu_zone`, `your_project_id` and `your_bucket` with your values. Note that the command below will only sample one query per passage. If you want multiple samples, you will need to repeat this process multiple times (remember to replace `output_filename` with a new filename for each sample).

```bash
for ITER in {00..08}; do
    t5_mesh_transformer \
      --tpu="your_tpu" \
      --gcp_project="your_project_id" \
      --tpu_zone="your_tpu_zone" \
      --model_dir="gs://your_bucket/models/" \
      --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
      --gin_file="infer.gin" \
      --gin_file="sample_decode.gin" \
      --gin_param="infer_checkpoint_step = 1004000" \
      --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 64}" \
      --gin_param="Bitransformer.decode.max_decode_length = 64" \
      --gin_param="input_filename = 'gs://your_bucket/data/input_docs.txt$ITER'" \
      --gin_param="output_filename = 'gs://your_bucket/data/predicted_queries_topk_sample.txt$ITER'" \
      --gin_param="tokens_per_batch = 131072" \
      --gin_param="Bitransformer.decode.temperature = 1.0" \
      --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = 10"
done
```
It should take approximately 8 hours to sample one query for each of the 8.8M passages, costing ~$20 USD (8 hours at $2.40 USD/hour) on a preemptible TPU.

## Learning a New Prediction Model: T5 Training with TensorFlow

Finally, we show how to learn a new prediction model.
The following command will train a T5-base model for 4k iterations to predict queries from passages.
We assume you put the tsv training file in `gs://your_bucket/data/doc_query_pairs.train.tsv` (download from above).
Also, change `your_tpu_name`, `your_tpu_zone`, `your_project_id`, and `your_bucket` accordingly.

```bash
t5_mesh_transformer  \
  --tpu="your_tpu_name" \
  --gcp_project="your_project_id" \
  --tpu_zone="your_tpu_zone" \
  --model_dir="gs://your_bucket/models/" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/base/model.ckpt-999900'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://your_bucket/data/doc_query_pairs.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1004000" \
  --gin_param="tokens_per_batch = 131072"
```

## Replicating MS MARCO Document Retrieval Results with Anserini

Here we detail how to expand documents from the MS MARCO _document_ dataset using docTTTTTquery.
The MS MARCO document dataset is similar to the MS MARCO passage, but it contains longer documents,
which need to be split into shorter segments before being fed to docTTTTTquery.

Like in the instructions for MS MARCO passage dataset, we explain the process in reverse order.
(i.e., indexing, expansion, query prediction), since we believe there are more users interested in
experimenting with the expanded index than expanding the document themselves.


### Indexing with expanded queries

First, download the original corpus, the predicted queries, and a file mapping document segments to their document id:
```
gsutil cp gs://neuralresearcher_data/msmarco_doc/msmarco-docs.tsv.gz .
gsutil cp gs://neuralresearcher_data/doc2query_t5_msmarco_doc/data/1/predicted_queries_topk_sample00?.txt???-1004000 .
gsutil cp gs://neuralresearcher_data/doc2query_t5_msmarco_doc/data/1/segment_doc_ids.txt .
```

Merge the predicted queries into a single file. There are 10 predicted queries per document.
```bash
for SAMPLE in {000..009}; do
    cat predicted_queries_topk_sample$SAMPLE.txt???-1004000 > predicted_queries_topk_sample$SAMPLE.txt-1004000
done

paste -d" " \
  predicted_queries_topk_sample000.txt-1004000 \
  predicted_queries_topk_sample001.txt-1004000 \
  predicted_queries_topk_sample002.txt-1004000 \
  predicted_queries_topk_sample003.txt-1004000 \
  predicted_queries_topk_sample004.txt-1004000 \
  predicted_queries_topk_sample005.txt-1004000 \
  predicted_queries_topk_sample006.txt-1004000 \
  predicted_queries_topk_sample007.txt-1004000 \
  predicted_queries_topk_sample008.txt-1004000 \
  predicted_queries_topk_sample009.txt-1004000 \
  > predicted_queries_topk.txt-1004000
```

We now append the queries to the original documents (this step takes approximately 10 minutes):
```bash
python convert_msmarco_doc_to_anserini.py \
  --original_docs_path=./msmarco-docs.tsv.gz \
  --doc_ids_path=./segment_doc_ids.txt \
  --predictions_path=./predicted_queries_topk.txt-1004000 \
  --output_docs_path=./expanded_docs/docs.json
```

Once we have the expanded document, we index them with Anserini (this step takes approximately 40 minutes):
```bash
sh ${PATH_TO_ANSERINI}/target/appassembler/bin/IndexCollection \
  -collection JsonCollection  \
  -generator DefaultLuceneDocumentGenerator \
  -input ./expanded_docs \
  -index ./lucene-index \
  -threads 6
```

We can then retrieve the documents using the dev queries (this step takes approximately 10 minutes).
```
sh ${PATH_TO_ANSERINI}/target/appassembler/bin/SearchCollection \
  -topicreader TsvString \
  -index ./lucene-index \
  -topics ${PATH_TO_ANSERINI}/src/main/resources/topics-and-qrels/topics.msmarco-doc.dev.txt \
  -output ./run.dev.small.txt \
  -bm25 \
  -threads 6
```

And evaluate using `trec_eval` tool:
```bash
${PATH_TO_ANSERINI}/eval/trec_eval.9.0.4/trec_eval \
  -m map -m recall.1000 \
  ${PATH_TO_ANSERINI}/src/main/resources/topics-and-qrels/qrels.msmarco-doc.dev.txt \
  ./run.dev.small.txt
```

The output should be:
```
map                   	all	0.2886
recall_1000           	all	0.9259
```

In comparison, indexing with the original documents gives:
```
map                     all     0.2310
recall_1000             all     0.8856
```

### Indexing with segmented documents and expanded queries

We split the documents into segments and append the expanded queries to each of them, then we create the index for them. In this way, we can keep more useful segments from the documents in the initial ranking stage. 

We will reuse the file `predicted_queries_topk.txt-1004000` that contains all the predicted queries from last section. We can now append the queries to the segmented documents.
```
python convert_segmented_msmarco_doc_to_anserini.py \
  --original_docs_path=./msmarco-docs.tsv.gz \
  --doc_ids_path=./segment_doc_ids.txt \
  --predictions_path=./predicted_queries_topk.txt-1004000 \
  --output_docs_path=./segmented_expanded_docs/docs.json
```

We index the segmented documents with Anserini:
```bash
sh ${PATH_TO_ANSERINI}/target/appassembler/bin/IndexCollection \
  -collection JsonCollection  \
  -generator DefaultLuceneDocumentGenerator \
  -input ./segmented_expanded_docs \
  -index ./segmented-docs-index \
  -threads 6
```

Then, we can retrieve the top 10k segments with dev queries:
```
sh ${PATH_TO_ANSERINI}/target/appassembler/bin/SearchCollection \
  -topicreader TsvString \
  -index ./segmented-docs-index \
  -topics ${PATH_TO_ANSERINI}/src/main/resources/topics-and-qrels/topics.msmarco-doc.dev.txt \
  -output ./run.dev.seg.small.txt \
  -hits 10000 \
  -bm25 \
  -threads 6
```

After that, we will aggregate the top 10000 segments into top 1000 document:
```
python convert_seg_to_doc.py
  --input ./run.dev.seg.small.txt
  --output ./run.dev.seg.top1k.small.txt
  --hits 1000
```

Finally, we can evaluate them.
```
bash
${PATH_TO_ANSERINI}/tools/eval/trec_eval.9.0.4/trec_eval \
  -m map -m recall.1000 \
  ${PATH_TO_ANSERINI}/src/main/resources/topics-and-qrels/qrels.msmarco-doc.dev.txt \
  ./run.dev.seg.top1k.small.txt
```

The output should be:
```
map                   	all	0.3182
recall_1000           	all	0.949
```

## Predicting Queries from Documents: T5 Inference with TensorFlow

If you want to predict the queries yourself, please follow the instructions below.

We begin by downloading the corpus, which contains 3.2M documents.
```bash
wget http://msmarco.blob.core.windows.net/msmarcoranking/msmarco-docs.tsv.gz
gunzip msmarco-docs.tsv.gz
```

We split the corpus into files of 100k documents, which later can be processed in parallel.
```bash
split --suffix-length 2 --numeric-suffixes --lines 100000 msmarco-docs.tsv msmarco-docs.tsv
```

We now segment each document using a sliding window of 10 sentences and stride of 5 sentences:
```bash
for ITER in {00..32}; do
    python convert_msmarco_doc_to_t5_format.py \
        --corpus_path=msmarco-docs.tsv$ITER \
        --output_segment_texts_path=${OUTPUT_DIR}/segment_texts.txt$ITER \
        --output_segment_doc_ids_path=${OUTPUT_DIR}/segment_doc_ids.txt$ITER
done
```

We are now ready to run inference. Since this is a costly step, we recommend using Google Cloud
with TPUs to run it faster.

We will use the docTTTTTquery model trained on the MS MARCO passage dataset, so you need to upload it to your Google Storage bucket.
```bash
wget https://storage.googleapis.com/doctttttquery_git/t5-base.zip
unzip t5-base.zip
gsutil cp model.ckpt-1004000* gs://your_bucket/models/
```

Run the command below to sample one question per segment (note that you will need to start a TPU).
```bash
for ITER in {00..32}; do
    t5_mesh_transformer \
      --tpu="your_tpu" \
      --gcp_project="your_project_id" \
      --tpu_zone="your_tpu_zone" \
      --model_dir="gs://your_bucket/models/" \
      --gin_file="gs://t5-data/pretrained_models/base/operative_config.gin" \
      --gin_file="infer.gin" \
      --gin_file="sample_decode.gin" \
      --gin_param="infer_checkpoint_step = 1004000" \
      --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 64}" \
      --gin_param="Bitransformer.decode.max_decode_length = 64" \
      --gin_param="input_filename = './segment_texts.txt$ITER'" \
      --gin_param="output_filename = './predicted_queries_topk_sample.txt$ITER'" \
      --gin_param="tokens_per_batch = 131072" \
      --gin_param="Bitransformer.decode.temperature = 1.0" \
      --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = 10"
done
```
