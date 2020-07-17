'''Converts MSMARCO's tsv collection to Anserini jsonl files with field configurations.'''
import argparse
import json
import os


# NLTK English stopwords
stop_words = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there',
              'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they',
              'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
              'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who',
              'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below',
              'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
              'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',
              'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she',
              'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and',
              'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then',
              'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not',
              'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
              'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
              'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
              'how', 'further', 'was', 'here', 'than'}


# process text by tokenizing and removing stopwords
def process_text(text):
    processed = text.lower().replace('.', ' ').replace(',', ' ').replace('?', ' ')
    return [word for word in processed.split() if word not in stop_words]


# split new and repeated prediction words
def split_new_repeated(pred_text, doc_text):
    pred_repeated = []
    pred_new = []

    doc_text_set = set(process_text(doc_text))
    processed_pred_text = process_text(pred_text)
    for word in processed_pred_text:
        if word in doc_text_set:
            pred_repeated.append(word)
        else:
            pred_new.append(word)

    return pred_new, pred_repeated


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

    # parameters to simulate BM25F via duplicated text
    parser.add_argument('--original_copies', default=1, type=int,
                        help='number of original text duplicates.')
    parser.add_argument('--prediction_copies', default=1, type=int,
                        help='number of predicted text duplicates.')

    # parameters to separate new and repeated prediction text
    parser.add_argument('--split_predictions', default=False, type=bool,
                        help='separate predicted text into repeated and new.')
    parser.add_argument('--repeated_prediction_copies', default=1, type=int,
                        help='number of repeated predicted text duplicates, \
                              must set split_predictions to true.')
    parser.add_argument('--new_prediction_copies', default=1, type=int,
                        help='number of new predicted text duplicates, \
                              must set split_predictions to true.')

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    print('Converting collection...')

    file_index = 0
    new_words = 0
    total_words = 0

    with open(args.collection_path) as f_corpus, \
            open(args.predictions) as f_pred:
        for i, (line_doc, line_pred) in enumerate(zip(f_corpus, f_pred)):
            # Write to a new file when the current one reaches maximum capacity.
            if i % args.max_docs_per_file == 0:
                if i > 0:
                    output_jsonl_file.close()
                output_path = os.path.join(
                    args.output_folder, 'docs{:02d}.json'.format(file_index))
                output_jsonl_file = open(output_path, 'w')
                file_index += 1

            doc_id, doc_text = line_doc.rstrip().split('\t')
            pred_text = line_pred.rstrip()

            contents = ''
            if args.split_predictions:
                pred_new, pred_repeated = split_new_repeated(pred_text, doc_text)
                new_words += len(pred_new)
                total_words += len(pred_new) + len(pred_repeated)

                contents += (doc_text + ' ') * args.original_copies
                contents += (' '.join(pred_repeated) + ' ') * args.repeated_prediction_copies
                contents += (' '.join(pred_new) + ' ') * args.new_prediction_copies
            else:
                contents += (doc_text + ' ') * args.original_copies
                contents += (pred_text + ' ') * args.prediction_copies

            output_dict = {'id': doc_id, 'contents': contents}
            output_jsonl_file.write(json.dumps(output_dict) + '\n')

            if i % 100000 == 0:
                print('Converted {} docs in {} files'.format(i, file_index))

    if args.split_predictions:
        print(f"Found {100 * new_words/total_words}% new predicted text")

    output_jsonl_file.close()
    print('Done!')
