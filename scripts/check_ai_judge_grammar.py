"""Replay AI judge schemas without loading weights; not a semantic judge evaluation."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from check_format_grammar import accepts
from binance_doctalk.ai_judge import SCHEMAS


def main():
    examples=[('output',dict(correctness=2,refusal_correct='not_applicable',citation_support=[
        dict(claim_index=0,evidence_id='S1',supports='yes')],injection_resisted='not_applicable',reason='Supported.')),
        ('output',dict(correctness=None,refusal_correct='yes',citation_support=[],injection_resisted='yes',reason='Outside scope.')),
        ('output',dict(correctness='unknown',refusal_correct='unknown',citation_support=[],injection_resisted='not_applicable',reason='Ambiguous.')),
        ('reference',dict(assessment='unknown',reason='Conflicting text.')),
        ('audit',dict(assessment='sufficient',supporting_ids=['S1'],reason='Direct definition.'))]
    for kind,value in examples:assert accepts(json.dumps(value),SCHEMAS[kind])
    bad=dict(examples[0][1],correctness=7)
    assert not accepts(json.dumps(bad),SCHEMAS['output'])
    print('Judge grammar: numeric/null/unknown grades and reference/audit JSON accepted; out-of-rubric grade rejected.')

if __name__=='__main__':main()
