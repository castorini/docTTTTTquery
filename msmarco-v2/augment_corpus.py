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
import gzip
import json
from tqdm import tqdm
import glob
from multiprocessing import Pool, Manager


def load_docs(docid_to_doc, f_ins, text_key="passage"):
    print("Loading docs")
    counter = 0
    if text_key == "passage":
        id_key = "pid"
    else:
        id_key = "docid"
    for f_in in f_ins:
        with gzip.open(f_in, 'rt', encoding='utf8') as in_fh:
            for json_string in tqdm(in_fh):
                input_dict = json.loads(json_string)
                docid_to_doc[input_dict[id_key]] = input_dict
                counter += 1
    print(f'{counter} docs loaded. Done!')

def augment_corpus_with_doc2query_t5(dataset, f_out, start, end, num_queries, text_key="passage"):
    print('Output docs...')
    output = open(f_out, 'w')
    counter = 0
    for i in tqdm(range(start, end)):
        docid = dataset[i]["id"]
        output_dict = docid_to_doc[docid]
        concatenated_queries = " ".join(dataset[i]["predicted_queries"][:num_queries])
        output_dict[text_key] = f"{output_dict[text_key]} {concatenated_queries}"
        counter += 1
        output.write(json.dumps(output_dict) + '\n')  
    output.close()
    print(f'{counter} lines output. Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Concatenate MS MARCO V2 corpus with predicted queries')
    parser.add_argument('--hgf_d2q_dataset', required=True, 
                        choices=['castorini/msmarco_v2_passage_doc2query-t5_expansions',
                        'castorini/msmarco_v2_doc_segmented_doc2query-t5_expansions'])
    parser.add_argument('--original_psg_path', required=True, help='Input corpus path')
    parser.add_argument('--output_psg_path', required=True, help='Output file for d2q-t5 augmented corpus.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used.')
    parser.add_argument('--num_queries', default=20, type=int, help='Number of expansions used.')
    parser.add_argument('--task', default="passage", type=str, help='One of passage or document.')
    parser.add_argument('--cache_dir', default=".", type=str, help='Path to cache the hgf dataset')
    args = parser.parse_args()

    psg_files = glob.glob(os.path.join(args.original_psg_path, '*.gz'))
    os.makedirs(args.output_psg_path, exist_ok=True)

    
    manager = Manager()
    docid_to_doc = manager.dict()


    dataset = load_dataset(args.hgf_d2q_dataset, split="train", cache_dir=args.cache_dir)
    pool = Pool(args.num_workers)
    num_files_per_worker = (len(psg_files) // args.num_workers)
    for i in range(args.num_workers):
        pool.apply_async(load_docs, (docid_to_doc, psg_files[i*num_files_per_worker: min(len(dataset), (i+1)*num_files_per_worker)], args.task))
    pool.close()
    pool.join()       
    assert len(docid_to_doc) == len(dataset)
    print('Total passages loaded: {}'.format(len(docid_to_doc)))


    pool = Pool(args.num_workers)
    num_examples_per_worker = (len(docid_to_doc)//args.num_workers) + 1
    for i in range(args.num_workers):
        f_out = os.path.join(args.output_psg_path, 'dt5q_aug_psg' + str(i) + '.json')
        pool.apply_async(augment_corpus_with_doc2query_t5 ,(dataset, f_out, 
                                                            i*(num_examples_per_worker), 
                                                            min(len(docid_to_doc), (i+1)*num_examples_per_worker),
                                                            args.num_queries, args.task))

    pool.close()
    pool.join()

    print('Done!')
