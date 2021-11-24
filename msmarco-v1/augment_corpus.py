#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
from datasets import load_dataset
import os
import json
from tqdm import tqdm
import glob
from pyserini.search import SimpleSearcher

def augment_corpus_with_doc2query_t5(dataset, searcher, f_out, num_queries, hgf_d2q_dataset, text_key="contents"):
    print('Output docs...')
    output = open(f_out, 'w')
    counter = 0
    for i in tqdm(range(len(dataset))):
        docid = dataset[i]["id"]
        try:
            if hgf_d2q_dataset == 'castorini/msmarco_v1_doc_doc2query-t5_expansions':
                split_sentences = searcher.doc(docid).raw().split('\n')
                output_dict = {}
                output_dict["id"] = docid
                text_content = " ".join(split_sentences[3:-2])
                output_dict["contents"] = f"{split_sentences[1]} {split_sentences[2]}. {text_content}"
            else:
                output_dict = json.loads(searcher.doc(docid).raw())
        except:
            print(f"{docid} not found in index")
            continue
        if num_queries == -1:
            concatenated_queries = " ".join(dataset[i]["predicted_queries"])
        else:
            concatenated_queries = " ".join(dataset[i]["predicted_queries"][:num_queries])
        output_dict[text_key] = f"{output_dict[text_key]} {concatenated_queries}"
        counter += 1
        output.write(json.dumps(output_dict) + '\n')  
    output.close()
    print(f'{counter} lines output. Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Concatenate MS MARCO V1 corpus with predicted queries')
    parser.add_argument('--hgf_d2q_dataset', required=True, 
                        choices=['castorini/msmarco_v1_passage_doc2query-t5_expansions',
                        'castorini/msmarco_v1_doc_segmented_doc2query-t5_expansions',
                        'castorini/msmarco_v1_doc_doc2query-t5_expansions'])
    parser.add_argument('--prebuilt_index', required=True, help='Prebuilt index name')
    parser.add_argument('--output_psg_path', required=True, help='Output file for d2q-t5 augmented corpus.')
    parser.add_argument('--num_queries', default=-1, type=int, help='Number of expansions used.')
    parser.add_argument('--cache_dir', default=".", type=str, help='Path to cache the hgf dataset')
    args = parser.parse_args()

    os.makedirs(args.output_psg_path, exist_ok=True)

    dataset = load_dataset(args.hgf_d2q_dataset, split="train", cache_dir=args.cache_dir)
    if args.prebuilt_index in ['msmarco-passage', 'msmarco-doc-per-passage', 'msmarco-doc']:
        searcher = SimpleSearcher.from_prebuilt_index(args.prebuilt_index)
    else:
        searcher = SimpleSearcher(args.prebuilt_index)
    augment_corpus_with_doc2query_t5(dataset, searcher, os.path.join(args.output_psg_path, "docs.jsonl"), args.num_queries, args.hgf_d2q_dataset)
    print('Done!')
