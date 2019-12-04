'''Converts MSMARCO's tsv collection to Anserini jsonl files.'''
import json
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Converts MSMARCO\'s tsv collection to Anserini jsonl '
                    'files.')
    parser.add_argument('--collection_path', required=True,
                        help='MS MARCO .tsv collection file')
    parser.add_argument('--predictions', required=True,
                        help='File containing predicted queries.')
    parser.add_argument('--output_folder', required=True, help='output folder')
    parser.add_argument('--max_docs_per_file', default=1000000, type=int,
                        help='maximum number of documents in each jsonl file.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    print('Converting collection...')

    file_index = 0
    with open(args.collection_path) as f_corpus, \
            open(args.predictions) as f_pred:
        for i, line_doc, line_pred in enumerate(zip(f_corpus, f_pred)):

            # Start writting to a new file whent the current one reached its
            # maximum capacity.
            if i % args.max_docs_per_file == 0:
                if i > 0:
                    output_jsonl_file.close()
                output_path = os.path.join(
                    args.output_folder, 'docs{:02d}.json'.format(file_index))
                output_jsonl_file = open(output_path, 'w')
                file_index += 1

            doc_id, doc_text = line_doc.rstrip().split('\t')
            pred_text = line_pred.rstrip()

            # Reads from predictions and merge then to the original doc text.
            text = doc_text + ' ' + pred_text

            output_dict = {'id': doc_id, 'contents': text}
            output_jsonl_file.write(json.dumps(output_dict) + '\n')

            if i % 100000 == 0:
                print('Converted {} docs in {} files'.format(i, file_index))

    output_jsonl_file.close()
    print('Done!')
