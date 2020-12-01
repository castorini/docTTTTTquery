# Document Expansion by Query Prediction

The repo describes experiments with docTTTTTquery (sometimes written as docT5query or doc2query-T5), the latest version of the doc2query family of document expansion models.
The basic idea is to train a model, that when given an input document, generates questions that the document might answer (or more broadly, queries for which the document might be relevant).
These predicted questions (or queries) are then appended to the original documents, which are then indexed as before.
The docTTTTTquery model gets its name from the use of T5 as the expansion model.

The primary advantage of this approach is that expensive neural inference is pushed to _indexing time_, which means that "bag of words" queries against an inverted index built on the augmented document collection are only slightly slower (due to longer documents) &mdash; but the retrieval results are _much_ better.
Of course, these documents can be further reranked by another neural model in a [multi-stage ranking architecture](https://arxiv.org/abs/1910.14424).

This technique was introduced in November 2019 on MS MARCO passage ranking task.
Results on the [leaderboard](https://microsoft.github.io/msmarco/) show that docTTTTTquery is much more effective than doc2query and (almost) as effective as the best non-BERT ranking model, while increasing query latency (time to retrieve 1000 docs per query) only slightly compared to vanilla BM25:

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

+ [Data and Trained Models: MS MARCO Passage Ranking Dataset](#Data-and-Trained-Models-MS-MARCO-Passage-Ranking-Dataset)
+ [Replicating MS MARCO Passage Ranking Results with Anserini](#Replicating-MS-MARCO-Passage-Ranking-Results-with-Anserini)
+ [Predicting Queries from Passages: T5 Inference with PyTorch](#Predicting-Queries-from-Passages-T5-Inference-with-PyTorch)
+ [Predicting Queries from Passages: T5 Inference with TensorFlow](#Predicting-Queries-from-Passages-T5-Inference-with-TensorFlow)
+ [Learning a New Prediction Model: T5 Training with TensorFlow](#Learning-a-New-Prediction-Model-T5-Training-with-TensorFlow)
+ [Replicating MS MARCO Document Ranking Results with Anserini](#Replicating-MS-MARCO-Document-Ranking-Results-with-Anserini)
+ [Predicting Queries from Documents: T5 Inference with TensorFlow](#Predicting-Queries-from-Documents-T5-Inference-with-TensorFlow)

## Data and Trained Models: MS MARCO Passage Ranking Dataset

The basic docTTTTTquery model is trained on the MS MARCO passage ranking dataset.
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
`doc_query_pairs.train.tsv` | 197 MB | `aa673014f93d43837ca4525b9a33422c` | [[Dropbox](https://www.dropbox.com/s/5i64irveqvvegey/doc_query_pairs.train.tsv?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/doc_query_pairs.train.tsv)]
`queries.dev.small.tsv` | 283 KB | `4621c583f1089d223db228a4f95a05d1` | [[Dropbox](https://www.dropbox.com/s/hq6xjhswiz60siu/queries.dev.small.tsv?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/queries.dev.small.tsv)]
`qrels.dev.small.tsv` | 140 KB| `38a80559a561707ac2ec0f150ecd1e8a` | [[Dropbox](https://www.dropbox.com/s/khsplt2fhqwjs0v/qrels.dev.small.tsv?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/qrels.dev.small.tsv)]
`collection.tar.gz` | 987 MB | `87dd01826da3e2ad45447ba5af577628` | [[Dropbox](https://www.dropbox.com/s/lvvpsx0cjk4vemv/collection.tar.gz?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/collection.tar.gz)]
`predicted_queries_topk_sampling.zip` | 7.9 GB | `8bb33ac317e76385d5047322db9b9c34` | [[Dropbox](https://www.dropbox.com/s/uzkvv4gpj3a596a/predicted_queries_topk_sampling.zip?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/predicted_queries_topk_sampling.zip)]
`run.dev.small.tsv` | 127 MB | `c7a2006ec92f1f25955a314acd9b81b0` | [[Dropbox](https://www.dropbox.com/s/nc1drdkjpxxsngg/run.dev.small.tsv?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/run.dev.small.tsv)]
`t5-base.zip` | 357 MB | `881d3ca87c307b3eac05fae855c79014` | [[Dropbox](https://www.dropbox.com/s/q1nye6wfsvf5sen/t5-base.zip?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip)]
`t5-large.zip` | 1.2 GB | `21c7e625210b0ae872679bc36ed92d44` | [[Dropbox](https://www.dropbox.com/s/gzq8r68uk38bmum/t5-large.zip?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-large.zip)]

## Replicating MS MARCO Passage Ranking Results with Anserini

We provide instructions on how to replicate our docTTTTTquery results for the MS MARCO passage ranking task with the [Anserini](https://github.com/castorini/anserini) IR toolkit, using the predicted queries provided above.

First, install Anserini (see [homepage](https://github.com/castorini/anserini) for more details):

```bash
sudo apt-get install maven
git clone --recurse-submodules https://github.com/castorini/anserini.git
cd anserini
mvn clean package appassembler:assemble
cd tools/eval && tar xvfz trec_eval.9.0.4.tar.gz && cd trec_eval.9.0.4 && make && cd ../../..
cd tools/eval/ndeval && make && cd ../../..
```

For the purposes of this of this guide, we'll assume that `anserini` is cloned as a sub-directory of this repo, i.e., `docTTTTTquery/anserini/`.

Next, download `queries.dev.small.tsv`, `qrels.dev.small.tsv`, `collection.tar.gz`, and `predicted_queries_topk_sampling.zip` using one of the options above.
The first three files can go into base directory of the repo `docTTTTTquery/`, but put the zip file in a separate sub-directory `docTTTTTquery/passage-predictions`.
The zip file contains a lot of individual files, so this will keep your directory structure manageable.

Before appending the predicted queries to the passages, we need to concatenate them.
The commands below create a file that contains 40 concatenated predictions per line and 8,841,823 lines, one for each passage in the corpus.
We concatenate only the first 40 predictions as there is only a tiny gain in MRR@10 when using all 80 predictions (nevertheless, we provide 80 predictions in case researchers want to use this data for other purposes).

```bash
cd passage-predictions

unzip predicted_queries_topk_sampling.zip

for i in $(seq -f "%03g" 0 17); do
    echo "Processing chunk $i"
    paste -d" " predicted_queries_topk_sample0[0-3]?.txt${i}-1004000 \
    > predicted_queries_topk.txt${i}-1004000
done

cat predicted_queries_topk.txt???-1004000 > predicted_queries_topk.txt-1004000
```

As a sanity check:

```bash
$ wc predicted_queries_topk.txt-1004000
 8841823 2253863941 12517353325 predicted_queries_topk.txt-1004000
```

Go back to your repo base directory `docTTTTTquery/`.
We can now append the predicted queries to the original MS MARCO passage collection:

```bash
tar xvf collection.tar.gz

python convert_msmarco_passage_to_anserini.py \
  --collection_path=collection.tsv \
  --predictions=passage-predictions/predicted_queries_topk.txt-1004000 \
  --output_folder=msmarco-passage-expanded
```

Now, create an index using Anserini on the expanded passages (we're assuming Anserini is cloned as a sub-directory):

```bash
sh anserini/target/appassembler/bin/IndexCollection \
  -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
  -threads 9 -input msmarco-passage-expanded -index lucene-index-msmarco-passage-expanded
```

Once the expanded passages are indexed, we can retrieve 1000 passages per query for the MS MARCO dev set:

```bash
sh anserini/target/appassembler/bin/SearchMsmarco \
  -index lucene-index-msmarco-passage-expanded -queries queries.dev.small.tsv \
  -output run.msmarco-passage-expanded.dev.small.txt -hits 1000 -threads 8
```

Finally, we evaluate the results using the MS MARCO eval script:

```bash
python anserini/tools/eval/msmarco_eval.py qrels.dev.small.tsv run.msmarco-passage-expanded.dev.small.txt
```

The results should be:

```
#####################
MRR @10: 0.27680089370991834
QueriesRanked: 6980
#####################
```

Voilà!

## Predicting Queries from Passages: T5 Inference with PyTorch

We will use the excellent [🤗 Transformers library](https://github.com/huggingface/transformers) by Hugging Face to sample queries from our T5 model.

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
wget https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip
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
  --gin_param="tokens_per_batch = 131072" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology ='v3-8'
```

## Replicating MS MARCO Document Ranking Results with Anserini

Here we detail how to replicate docTTTTTquery runs for the MS MARCO _document_ ranking task.
The MS MARCO document ranking task is similar to the MS MARCO passage ranking task, but the corpus contains longer documents, which need to be split into shorter passages before being fed to docTTTTTquery.

Like the instructions for MS MARCO passage ranking task, we explain the process in reverse order (i.e., indexing, expansion, query prediction), since we believe there are more users interested in experimenting with the expanded index than expanding the document themselves.

Here are the relevant files to download:

File | Size | MD5 | Download
:----|-----:|:----|:-----
`msmarco-docs.tsv.gz` | 7.9 GB | `103b19e21ad324d8a5f1ab562425c0b4` | [[Dropbox](https://www.dropbox.com/s/t7r324wchnf98pm/msmarco-docs.tsv.gz?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-doc/msmarco-docs.tsv.gz)]
`predicted_queries_doc.tar.gz` | 2.2 GB | `4967214dfffbd33722837533c838143d` | [[Dropbox](https://www.dropbox.com/s/s4vwuampddu7677/predicted_queries_doc.tar.gz?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-doc/predicted_queries_doc.tar.gz)]
`msmarco_doc_passage_ids.txt` | 170 MB | `82c00bebab0d98c1dc07d78fac3d8b8d` | [[Dropbox](https://www.dropbox.com/s/wi6i2hzkcmbmusq/msmarco_doc_passage_ids.txt?dl=1)] [[GitLab](https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-doc/msmarco_doc_passage_ids.txt)]

### Per-Document Expansion

The most straightforward way to use docTTTTTquery is to append the expanded queries to _each_ document.
First, download the original corpus (`msmarco-docs.tsv.gz`), the predicted queries (`predicted_queries_doc.tar.gz`), and a file mapping passages to their document ids (`msmarco_doc_passage_ids.txt`), using one of the options above.
Put `predicted_queries_doc.tar.gz` in a sub-directory `doc-predictions/`.

Merge the predicted queries into a single file; there are 10 predicted queries per document.
This can be accomplished as follows:

```bash
cd doc-predictions/

tar xvfz predicted_queries_doc.tar.gz

for i in $(seq -f "%03g" 0 9); do
    cat predicted_queries_doc_sample${i}.txt???-1004000 > predicted_queries_doc_sample${i}_all.txt
done

paste -d" " \
  predicted_queries_doc_sample000_all.txt \
  predicted_queries_doc_sample001_all.txt \
  predicted_queries_doc_sample002_all.txt \
  predicted_queries_doc_sample003_all.txt \
  predicted_queries_doc_sample004_all.txt \
  predicted_queries_doc_sample005_all.txt \
  predicted_queries_doc_sample006_all.txt \
  predicted_queries_doc_sample007_all.txt \
  predicted_queries_doc_sample008_all.txt \
  predicted_queries_doc_sample009_all.txt \
   > predicted_queries_doc_sample_all.txt
```

Sanity check:

```bash
$ md5sum predicted_queries_doc_sample_all.txt 
b01b2fbbb8d382684a80fbf51efbca93  predicted_queries_doc_sample_all.txt
$ wc predicted_queries_doc_sample_all.txt 
  20545677 1379262573 7672087649 predicted_queries_doc_sample_all.txt
```

We now append the queries to the original documents (this step takes approximately 10 minutes, the counter needs to get to 20545677):

```bash
python convert_msmarco_doc_to_anserini.py \
  --original_docs_path=msmarco-docs.tsv.gz \
  --doc_ids_path=msmarco_doc_passage_ids.txt \
  --predictions_path=doc-predictions/predicted_queries_doc_sample_all.txt \
  --output_docs_path=msmarco-doc-expanded/docs.json
```

Once we have the expanded documents (about 29 GB in size), the next step is to build an index with Anserini.
As above, we'll assume that Anserini is cloned as a sub-directory of this repo, i.e., `docTTTTTquery/anserini/`.
This step takes approximately 40 minutes:

```bash
sh anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator -threads 1 \
  -input msmarco-doc-expanded -index lucene-index-msmarco-doc-expanded
```

We can then retrieve the documents using the dev queries (this step takes approximately 10 minutes):

```
sh anserini/target/appassembler/bin/SearchCollection \
  -index lucene-index-msmarco-doc-expanded \
  -topicreader TsvString -topics anserini/src/main/resources/topics-and-qrels/topics.msmarco-doc.dev.txt \
  -output run.msmarco-doc-expanded.dev.small.txt -bm25
```

And evaluate using `trec_eval` tool:

```bash
anserini/tools/eval/trec_eval.9.0.4/trec_eval -m map -m recall.1000 \
  anserini/src/main/resources/topics-and-qrels/qrels.msmarco-doc.dev.txt run.msmarco-doc-expanded.dev.small.txt
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

### Per-Passage Expansion

Although per-document expansion is the most straightforward way to use docTTTTTquery, we have found that _per passage_ expansion works even better.
In this approach, we split the documents into passages and append the expanded queries to _each_ passage.
We then index the passages of this expanded corpus.

We will reuse the file `predicted_queries_doc_sample_all.txt` that contains all the predicted queries from last section.
To start, append the queries to the passages:

```
python convert_msmarco_passages_doc_to_anserini.py \
  --original_docs_path=msmarco-docs.tsv.gz \
  --doc_ids_path=msmarco_doc_passage_ids.txt \
  --predictions_path=doc-predictions/predicted_queries_doc_sample_all.txt \
  --output_docs_path=msmarco-doc-expanded-passage/docs.json
```

This step takes several hours (the counter needs to get to 20545677).
Upon completion, index the passages with Anserini:

```bash
sh anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator -threads 1 \
  -input msmarco-doc-expanded-passage -index lucene-index-msmarco-doc-expanded-passage
```

Then, we can retrieve the top 1k passages with dev queries:

```
sh anserini/target/appassembler/bin/SearchCollection \
  -index lucene-index-msmarco-doc-expanded-passage \
  -topicreader TsvString -topics anserini/src/main/resources/topics-and-qrels/topics.msmarco-doc.dev.txt \
  -output run.msmarco-doc-expanded-passage.dev.small.txt \
  -bm25 -hits 10000 -selectMaxPassage -selectMaxPassage.delimiter "#" -selectMaxPassage.hits 1000
```

In a bit more detail, we retrieve the top 10k passages per query, but then use Anserini's `-selectMaxPassage` option to select only the best (highest-scoring) passage from each document, finally returning top 1k docid per query.

Evaluation:

```
anserini/tools/eval/trec_eval.9.0.4/trec_eval -m map -m recall.1000 \
  anserini/src/main/resources/topics-and-qrels/qrels.msmarco-doc.dev.txt \
  run.msmarco-doc-expanded-passage.dev.small.txt
```

The output should be:

```
map                   	all	0.3182
recall_1000           	all	0.9490
```

In comparison with per-passage expansion, we will use per passage _without expansion_ as the baseline. In this method, we will not append the predicted queries to the passages.

We will first split the original documents into passages:
```
python convert_msmarco_passages_doc_to_anserini.py \
  --original_docs_path=msmarco-docs.tsv.gz \
  --doc_ids_path=msmarco_doc_passage_ids.txt \
  --output_docs_path=msmarco-doc-passage/docs.json \
```

It will also take several hours, and the generated file will be 27G. Same as what we did for per-passage expansion, we will use Anserini to index the file, retrieve the top1k passages from them for the dev queries and evaluate them.

```
sh anserini/target/appassembler/bin/IndexCollection -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator -threads 1 \
  -input msmarco-doc-passage -index lucene-index-msmarco-doc-passage

sh anserini/target/appassembler/bin/SearchCollection \
  -index lucene-index-msmarco-doc-passage \
  -topicreader TsvString -topics anserini/src/main/resources/topics-and-qrels/topics.msmarco-doc.dev.txt \
  -output run.msmarco-doc-passage.dev.small.txt \
  -bm25 -hits 10000 -selectMaxPassage -selectMaxPassage.delimiter "#" -selectMaxPassage.hits 1000

anserini/tools/eval/trec_eval.9.0.4/trec_eval -m map -m recall.1000 \
  anserini/src/main/resources/topics-and-qrels/qrels.msmarco-doc.dev.txt \
  run.msmarco-doc-passage.dev.small.txt
```

The result is:
```
map                   	all	0.2688
recall_1000           	all	0.9180
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
        --output_passage_texts_path=${OUTPUT_DIR}/passage_texts.txt$ITER \
        --output_passage_doc_ids_path=${OUTPUT_DIR}/msmarco_doc_passage_ids.txt$ITER
done
```

We are now ready to run inference. Since this is a costly step, we recommend using Google Cloud
with TPUs to run it faster.

We will use the docTTTTTquery model trained on the MS MARCO passage ranking dataset, so you need to upload it to your Google Storage bucket.
```bash
wget https://git.uwaterloo.ca/jimmylin/doc2query-data/raw/master/T5-passage/t5-base.zip
unzip t5-base.zip
gsutil cp model.ckpt-1004000* gs://your_bucket/models/
```

Run the command below to sample one question per passage (note that you will need to start a TPU).

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
      --gin_param="input_filename = './passage_texts.txt$ITER'" \
      --gin_param="output_filename = './predicted_queries_topk_sample.txt$ITER'" \
      --gin_param="tokens_per_batch = 131072" \
      --gin_param="Bitransformer.decode.temperature = 1.0" \
      --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = 10" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology ='v3-8'
done
```
