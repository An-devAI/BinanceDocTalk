"""Replay real failed outputs against the local decoding grammar, without GPU inference."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.local_provider import local_answer_schema, ABSTENTION_MESSAGES
from binance_doctalk.schemas import validate_answer
from lmformatenforcer import JsonSchemaParser, CharacterLevelParserConfig


def accepts(text, schema):
    parser=JsonSchemaParser(schema,config=CharacterLevelParserConfig(force_json_field_order=True))
    for char in text:
        if char not in parser.get_allowed_characters():return False
        parser=parser.add_character(char)
    return parser.can_end()


def main():
    fixtures=json.loads((ROOT/'tests/fixtures/local-format-failures.json').read_text())
    ids=['DEMO01:c0001','DEMO02:c0001']
    for case in fixtures['cases']:
        schema=local_answer_schema(case['mode']=='rag',ids)
        assert not accepts(case['raw_output'],schema), 'A captured invalid output is permitted by grammar'
    examples=[
        (True,{'status':'answered','reason':None,'message':'','claims':[
            {'text':'A limit order can remain unfilled.','evidence_ids':['DEMO01:c0001']}]}),
        (False,{'status':'answered','reason':None,'message':'','claims':[
            {'text':'A limit order can remain unfilled.','evidence_ids':[]}]}),
        (True,{'status':'abstained','reason':'out_of_scope','message':ABSTENTION_MESSAGES['out_of_scope'],'claims':[]}),
        (True,{'status':'abstained','reason':'insufficient_evidence','message':ABSTENTION_MESSAGES['insufficient_evidence'],'claims':[]}),
    ]
    for grounded,answer in examples:
        schema=local_answer_schema(grounded,ids)
        assert accepts(json.dumps(answer),schema), 'Valid answer rejected by grammar'
        validate_answer(answer,set(ids),grounded)
    invented={'status':'answered','reason':None,'message':'','claims':[
        {'text':'An invented citation.','evidence_ids':['INVENTED']}]}
    assert not accepts(json.dumps(invented),local_answer_schema(True,ids))
    hidden_answer={'status':'abstained','reason':'insufficient_evidence','message':'A factual answer hidden outside claims.','claims':[]}
    assert not accepts(json.dumps(hidden_answer),local_answer_schema(True,ids)), 'Uncited prose can hide in abstention message'
    assert accepts(json.dumps(examples[-1][1]),local_answer_schema(True,[]))
    assert not accepts(json.dumps(examples[0][1]),local_answer_schema(True,[]))
    print('Grammar replay: 4 captured failures blocked; 4 valid outputs accepted; invented IDs and uncited abstention prose blocked.')


if __name__=='__main__':main()
