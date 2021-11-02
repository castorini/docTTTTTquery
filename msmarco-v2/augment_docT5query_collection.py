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
import os
import sys
import gzip
import json
from tqdm import tqdm
import re
import glob
from multiprocessing import Pool, Manager


def read_queries(f_ins, docid_to_queries, num_queries):
    for f_in in f_ins:
        print('read {}...'.format(f_in))
        with gzip.open(f_in, 'rt', encoding='utf8') as in_fh:
            for json_string in tqdm(in_fh):
                queries_dict = json.loads(json_string)
                docid_to_queries[queries_dict["id"]] = " ".join(queries_dict["predicted_queries"][:num_queries])


def augment_corpus_with_doct5query(f_ins, f_out, text_key="passage"):
    print('Output passages...')
    output = open(f_out, 'w')
    counter = 0
    if text_key == "passage":
        id_key = "pid"
    else:
        id_key = "docid"
    for f_in in f_ins:
        with gzip.open(f_in, 'rt', encoding='utf8') as in_fh:
            for json_string in tqdm(in_fh):
                input_dict = json.loads(json_string)
                output_dict = input_dict
                counter+=1
                output_dict[text_key] = f"{output_dict[text_key]} {docid_to_queries[output_dict[id_key]]}"
                output.write(json.dumps(output_dict) + '\n')  
    output.close()
    print(f'{counter} lines output. Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Concatenate MS MARCO original docs with predicted queries')
    parser.add_argument('--input_d2q_path', required=True, help='Output file in the anserini jsonl format.')
    parser.add_argument('--original_psg_path', required=True, help='Json corpus file path.')
    parser.add_argument('--output_psg_path', required=True, help='Output file in the anserini jsonl format.')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of workers used.')
    parser.add_argument('--num_queries', default=20, type=int, help='Number of expansions used.')
    parser.add_argument('--task', default="passage", type=str, help='One of passage or document.')
    args = parser.parse_args()

    query_files = glob.glob(os.path.join(args.input_d2q_path, '*.gz'))
    psg_files = glob.glob(os.path.join(args.original_psg_path, '*.gz'))
    os.makedirs(args.output_psg_path, exist_ok=True)

    
    manager = Manager()
    docid_to_queries = manager.dict()

    num_files = len(query_files)
    pool = Pool(args.num_workers)
    num_files_per_worker=num_files//args.num_workers
    for i in range(args.num_workers):
        if i==(args.num_workers-1):
            file_list = query_files[i*num_files_per_worker:]
        else:
            file_list = query_files[i*num_files_per_worker:((i+1)*num_files_per_worker)]
        pool.apply_async(read_queries ,(file_list, docid_to_queries, args.num_queries))
    pool.close()
    pool.join() 
    print('Total passages queries loaded: {}'.format(len(docid_to_queries)))


    num_files = len(psg_files)
    pool = Pool(args.num_workers)
    num_files_per_worker=num_files//args.num_workers
    for i in range(args.num_workers):
        f_out = os.path.join(args.output_psg_path, 'dt5q_aug_psg' + str(i) + '.json')
        if i==(args.num_workers-1):
            file_list = psg_files[i*num_files_per_worker:]
        else:
            file_list = psg_files[i*num_files_per_worker:((i+1)*num_files_per_worker)]

        pool.apply_async(augment_corpus_with_doct5query ,(file_list, f_out, args.task))

    pool.close()
    pool.join()

    print('Done!')
