# docTTTTTquery

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
