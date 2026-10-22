"""Flag cross-split lexical overlaps for human review; not semantic deduplication."""
from __future__ import annotations
import argparse
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.evaluation import validate_questions
from binance_doctalk.io import read_jsonl, utc_now, write_json
from binance_doctalk.retrieval import terms


def audit(directory: Path):
    rows = [r for split in ['smoke','dev','test'] for r in read_jsonl(directory/f'{split}.jsonl')]
    structural = validate_questions(rows)
    pairs = []
    for a,b in combinations(rows,2):
        if a['split']==b['split']: continue
        wa,wb = set(terms(a['question'])),set(terms(b['question']))
        jaccard = len(wa&wb)/len(wa|wb) if wa|wb else 0
        sequence = SequenceMatcher(None,a['question'].casefold(),b['question'].casefold()).ratio()
        score = max(jaccard,sequence)
        pairs.append({'left_id':a['question_id'],'left_split':a['split'],'left_question':a['question'],
                          'right_id':b['question_id'],'right_split':b['split'],'right_question':b['question'],
                          'token_jaccard':jaccard,'character_sequence_similarity':sequence,
                          'priority_score':score,'above_flag_threshold':score>=.6,'human_decision':None})
    pairs.sort(key=lambda p:(-p['priority_score'],p['left_id'],p['right_id']))
    return {'created_at':utc_now(),'structural_audit':structural,'flag_threshold':.6,
            'flagged_cross_split_pairs':sum(p['above_flag_threshold'] for p in pairs),'candidates':pairs[:20],
            'status':'pending_human_semantic_review',
            'limitation':'Lexical heuristics can miss paraphrases and flag distinct questions sharing terminology. No case is automatically approved, removed, or reassigned.'}


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'data/eval/drafts')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Output exists; preserve previous review decisions')
    report = audit(args.directory)
    write_json(args.output,report)
    print({'questions':report['structural_audit']['questions'],
           'flagged_pairs':report['flagged_cross_split_pairs'],'status':report['status']})
