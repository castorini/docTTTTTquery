# docTTTTTquery

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
