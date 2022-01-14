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
from multiprocessing import Pool
from pyserini.search import SimpleSearcher


def augment_corpus_with_doc2query_t5(dataset, f_out, start, end, num_queries, text_key="passage"):
    print('Output docs...')
    output = open(f_out, 'w')
    counter = 0
    for i in tqdm(range(start, end)):
        docid = dataset[i]["id"]
        output_dict = {} #json.loads(index.doc(docid).raw())
        output_dict[text_key] = text_key
        if num_queries == -1:
            concatenated_queries = " ".join(dataset[i]["predicted_queries"])
        else:
            concatenated_queries = " ".join(dataset[i]["predicted_queries"][:num_queries])
        output_dict[text_key] = output_dict[text_key].replace("\n", " ")
        output_dict[text_key] = f"{output_dict[text_key]}\n{concatenated_queries}"
        counter += 1
        output.write(json.dumps(output_dict) + '\n')
    output.close()
    print(f'{counter} lines output. Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Concatenate MS MARCO V2 corpus with predicted queries')
    parser.add_argument('--hgf_d2q_dataset', required=True,
                        choices=['castorini/msmarco_v2_passage_doc2query-t5_expansions',
                                 'castorini/msmarco_v2_doc_segmented_doc2query-t5_expansions',
                                 'castorini/msmarco_v2_doc_doc2query-t5_expansions'])
    parser.add_argument('--index_path', required=True, help='Input index path')
    parser.add_argument('--output_psg_path', required=True, help='Output file for d2q-t5 augmented corpus.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used.')
    parser.add_argument('--num_queries', default=-1, type=int, help='Number of expansions used.')
    parser.add_argument('--cache_dir', default=".", type=str, help='Path to cache the hgf dataset')
    parser.add_argument('--task', default="passage", type=str, help='One of passage or document.')
    args = parser.parse_args()

    os.makedirs(args.output_psg_path, exist_ok=True)
    dataset = load_dataset(args.hgf_d2q_dataset, split="train", cache_dir=args.cache_dir)
    if args.index_path in ['msmarco-v2-passage', 'msmarco-v2-passage-augmented',
                           'msmarco-v2-doc-segmented', 'msmarco-v2-doc']:
        searcher = SimpleSearcher.from_prebuilt_index(args.index_path)
    else:
        searcher = SimpleSearcher(args.index_path)
    if searcher.num_docs != len(dataset):
        print("Total number of expanded queries: {}".format(len(dataset)))
    print('Total passages loaded: {}'.format(searcher.num_docs))
    with Pool(args.num_workers) as pool:
        for i in range(args.num_workers):
            f_out = os.path.join(args.output_psg_path, 'dt5q_aug_psg' + str(i) + '.json')
            print(f_out)
            start = i * (searcher.num_docs // args.num_workers)
            end = (i + 1) * (searcher.num_docs // args.num_workers)
            if i == args.num_workers - 1:
                end = searcher.num_docs
            pool.apply_async(augment_corpus_with_doc2query_t5,
                             args=(dataset, f_out, start, end, args.num_queries, args.task, ))
        pool.close()
        pool.join()

        print('Done!')
    print(f'{searcher.num_docs} documents and {len(dataset)} expanded documents.')
