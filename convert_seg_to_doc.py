'''
Aggregate the doc_segment_id in the run file into doc_id for evaluation. 

Input: trec run file for segments
Output: trec run file for documents
'''

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='input file path, expected format: query_id, Q0, doc_seg_id, rank, score, run_tag')
parser.add_argument('--output', required=True, help='output file path, expected format: query_id, Q0, doc_id, rank, score, run_tag')
parser.add_argument('--hits', required=True, help='max number of top documents to return')

args = parser.parse_args()

with open(args.input, 'r') as f:
    with open(args.output, 'w') as fout:
        query_ref = "query_ref"
        for line in f:
            query_id, a, doc_seg_id, rank, score, run_tag = line.split(' ')
            if query_id != query_ref:
                top_K = set()
                rank = 1
            if rank <= args.hits:
                doc_seg = doc_seg_id.split('#') 
                doc_id = doc_seg[0]
                if doc_id not in top_K:
                    fout.write(f"{query_id} {a} {doc_id} {rank} {score} {run_tag}")
                    top_K.add(doc_id)
                    rank += 1
            query_ref = query_id
                
                
