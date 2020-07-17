import argparse
import gzip
import json
import os

from tqdm import tqdm


parser = argparse.ArgumentParser(
    description='Concatenate MS MARCO original docs with predicted queries')
parser.add_argument('--original_docs_path', required=True)
parser.add_argument('--doc_ids_path', required=True)
parser.add_argument('--predictions_path', required=True)
parser.add_argument('--output_docs_path', required=True)

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
            predicted_queries = ' '.join(predicted_queries)
            expanded_text = f'{doc_url} {doc_title} {doc_text} {predicted_queries}'
            output_dict = {'id': doc_id, 'contents': expanded_text}
            f_out.write(json.dumps(output_dict) + '\n')

        doc_id, doc_url, doc_title, doc_text = next(f_corpus).split('\t')
        doc_text = doc_text.strip()
        predicted_queries = []

    predicted_queries.append(predicted_queries_partial)

f_corpus.close()
f_out.close()
print('Done!')
