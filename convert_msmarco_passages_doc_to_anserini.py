'''
Segment the documents and append their url, title, predicted queries to them. Then, they are saved into 
json which can be used for indexing.
'''

import argparse
import gzip
import json
import os
import spacy
from tqdm import tqdm
import re

def create_segments(doc_text, max_length, stride):
    doc_text = doc_text.strip()
    doc = nlp(doc_text[:10000])
    sentences = [sent.string.strip() for sent in doc.sents]
    segments = []
    
    for i in range(0, len(sentences), stride):
        segment = " ".join(sentences[i:i+max_length])
        segments.append(segment)
        if i + max_length >= len(sentences):
            break
    return segments

parser = argparse.ArgumentParser(
    description='Concatenate MS MARCO original docs with predicted queries')
parser.add_argument('--original_docs_path', required=True, help='MS MARCO .tsv corpus file.')
parser.add_argument('--doc_ids_path', required=True, help='File mapping segments to doc ids.')
parser.add_argument('--output_docs_path', required=True, help='Output file in the anserini jsonl format.')
parser.add_argument('--predictions_path', help='File containing predicted queries.')
parser.add_argument('--no_expansion', default=False, type=bool, help='expand with predicted queries or not')
parser.add_argument('--max_length', default=10)
parser.add_argument('--stride', default=5)
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_docs_path), exist_ok=True)

f_corpus = gzip.open(args.original_docs_path, mode='rt')
f_out = open(args.output_docs_path, 'w')
max_length = args.max_length
stride = args.stride
nlp = spacy.blank("en")
nlp.add_pipe(nlp.create_pipe("sentencizer"))

print('Spliting documents...')
doc_id_ref = None
if args.no_expansion:
    doc_ids_queries = zip(open(args.doc_ids_path))
elif not args.no_expansion:
    doc_ids_queries = zip(open(args.doc_ids_path),open(args.predictions_path))
for doc_id_query in tqdm(doc_ids_queries):
    doc_id = doc_id_query[0].strip()
    if doc_id != doc_id_ref:
        f_doc_id, doc_url, doc_title, doc_text = next(f_corpus).split('\t')
        while f_doc_id != doc_id:
            f_doc_id, doc_url, doc_title, doc_text = next(f_corpus).split('\t')
        segments = create_segments(doc_text, args.max_length, args.stride)
        seg_id = 0
    else:
        seg_id += 1
    doc_seg = f'{doc_id}#{seg_id}'
    if seg_id < len(segments):
        segment = segments[seg_id]
        if args.no_expansion:
            expanded_text = f'{doc_url} {doc_title} {segment}'
        elif not args.no_expansion:
            predicted_queries_partial = doc_id_query[1]
            expanded_text = f'{doc_url} {doc_title} {segment} {predicted_queries_partial}'
        output_dict = {'id': doc_seg, 'contents': expanded_text}
        f_out.write(json.dumps(output_dict) + '\n')  
    doc_id_ref = doc_id  
    
f_corpus.close()
f_out.close()
print('Done!')