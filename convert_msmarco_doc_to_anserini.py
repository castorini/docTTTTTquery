import argparse
import gzip
import json
import os

from tqdm import tqdm

def generate_output_dict(doc, predicted_queries):
    doc_id, doc_url, doc_title, doc_text = doc[0], doc[1], doc[2], doc[3]
    doc_text = doc_text.strip()
    predicted_queries = ' '.join(predicted_queries)
    expanded_text = f'{doc_url} {doc_title} {doc_text} {predicted_queries}'
    output_dict = {'id': doc_id, 'contents': expanded_text}
    return output_dict

parser = argparse.ArgumentParser(
    description='Concatenate MS MARCO original docs with predicted queries')
parser.add_argument('--original_docs_path', required=True, help='MS MARCO .tsv corpus file.')
parser.add_argument('--doc_ids_path', required=True, help='File mapping segments to doc ids.')
parser.add_argument('--predictions_path', required=True, help='File containing predicted queries.')
parser.add_argument('--output_docs_path', required=True, help='Output file in the anserini jsonl format.')

args = parser.parse_args()

os.makedirs(os.path.dirname(args.output_docs_path), exist_ok=True)

f_corpus = gzip.open(args.original_docs_path, mode='rt')
f_out = open(args.output_docs_path, 'w')

print('Appending predictions...')
doc_id = None
for doc_id_ref, predicted_queries_partial in tqdm(zip(open(args.doc_ids_path),
                                                      open(args.predictions_path))):
    doc_id_ref = doc_id_ref.strip()
    if doc_id_ref != doc_id:
        if doc_id is not None:
            output_dict = generate_output_dict(doc, predicted_queries)
            f_out.write(json.dumps(output_dict) + '\n')

        doc = next(f_corpus).split('\t')
        doc_id = doc[0]
        predicted_queries = []

    predicted_queries.append(predicted_queries_partial)

output_dict = generate_output_dict(doc, predicted_queries)
f_out.write(json.dumps(output_dict) + '\n')
f_corpus.close()
f_out.close()
print('Done!')
