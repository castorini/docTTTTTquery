import argparse
import re
import spacy
from tqdm import tqdm


def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    for line in tqdm(open(path)):
        doc_id, doc_url, doc_title, doc_text = line.split('\t')
        doc_text = doc_text.strip()
        corpus[doc_id] = (doc_title, doc_text)
    return corpus


parser = argparse.ArgumentParser(
    description='Create T5-formatted tsv file from MS MARCO Document Ranking '
                'dataset.')
parser.add_argument('--corpus_path', required=True, default='', help='')
parser.add_argument('--output_segment_texts_path', required=True, default='',
                    help='')
parser.add_argument('--output_segment_doc_ids_path', required=True, default='',
                    help='')
parser.add_argument('--stride', default=5, help='')
parser.add_argument('--max_length', default=10, help='')

args = parser.parse_args()

nlp = spacy.blank("en")
nlp.add_pipe(nlp.create_pipe("sentencizer"))

corpus = load_corpus(path=args.corpus_path)

n_segments = 0
n_no_segments = 0
with open(args.output_segment_texts_path, 'w') as fout_segment_texts, \
        open(args.output_segment_doc_ids_path, 'w') as fout_segment_doc_ids:

    for doc_id, (doc_title, doc_text) in tqdm(corpus.items(), total=len(corpus)):
        doc = nlp(doc_text[:10000])
        sentences = [sent.string.strip() for sent in doc.sents]
        if not sentences:
            n_no_segments += 1
        for i in range(0, len(sentences), args.stride):
            segment = ' '.join(sentences[i:i + args.max_length])
            segment = doc_title + ' ' + segment

            # Remove starting #'s as T5 skips those lines by default.
            segment = re.sub(r'^#*', '', segment)

            fout_segment_doc_ids.write(f'{doc_id}\n')
            fout_segment_texts.write(f'{segment}\n')
            n_segments += 1
            if i + args.max_length >= len(sentences):
                break

print(f'Wrote {n_segments} segments from {len(corpus)} docs.')
print(f'There were {n_no_segments} docs without segments/sentences.')

print('Done!')
