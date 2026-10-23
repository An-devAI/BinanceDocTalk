"""Prepare assistant-authored development annotations, not human-approved gold."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from binance_doctalk.benchmark import corpus_fingerprint,scope_policy,validate_bound_evidence
from binance_doctalk.evaluation import validate_questions
from binance_doctalk.ingest import load_documents
from binance_doctalk.io import digest,file_hash,read_jsonl,utc_now,write_json,write_jsonl
from prepare_benchmark_v2 import census


def validate_proposal(rows,docs):
    result=validate_questions(rows)
    if len(rows)!=36 or any(r['split']!='dev' for r in rows):
        raise ValueError('Exactly 36 development proposals are required; held-out is forbidden')
    by_id={d.doc_id:d.content_hash for d in docs}
    for r in rows:
        if r.get('annotation_author')!='AI assistant' or r.get('review_status')!='ai_annotated_pending_human_review':
            raise ValueError('Proposals must retain explicit AI authorship and pending human review')
        if r.get('reviewer') is not None or r.get('reviewed_at') is not None:
            raise ValueError('Do not assign a human reviewer or approval timestamp to an AI proposal')
        validate_bound_evidence(r,docs)
        if r['label']=='out_of_scope' or r['test_type']=='prompt_injection':
            if r['policy_basis']['sha256']!=scope_policy()['sha256']:
                raise ValueError('Proposal scope policy is stale')
        if r['label']=='insufficient_information':
            audit=r['corpus_audit'];entries=audit['documents']
            if (audit['corpus_hash']!=corpus_fingerprint(docs) or len(entries)!=len(by_id)
                    or {e['doc_id']:e['content_hash'] for e in entries}!=by_id):
                raise ValueError('Provisional gap audit must enumerate the exact whole corpus')
            if audit.get('review_status')!='ai_annotated_pending_human_review':
                raise ValueError('A provisional gap audit cannot claim human approval')
            if any(e.get('assessment') not in {'absent','partial'} or not e.get('notes') for e in entries):
                raise ValueError('Every provisional audit entry needs an assessment and rationale')
    return result


def prepare(corpus:Path,parent:Path,output:Path,changes_output:Path):
    if output.exists() or changes_output.exists():raise ValueError('Proposal version exists; preserve it')
    docs=load_documents(corpus);by_id={d.doc_id:d for d in docs}
    originals=read_jsonl(parent);rows=deepcopy(originals)
    def ref(doc_id,pid,fragment=None):
        doc=by_id[doc_id];p=next(p for p in doc.paragraphs if p.paragraph_id==pid)
        start=p.text.index(fragment) if fragment else 0
        end=start+len(fragment) if fragment else len(p.text)
        return dict(doc_id=doc_id,paragraph_id=pid,start=start,end=end,content_hash=doc.content_hash)
    def sentence(doc_id,pid,number=0):
        text=next(p.text for p in by_id[doc_id].paragraphs if p.paragraph_id==pid)
        pieces=[m.group().strip() for m in re.finditer(r'.+?(?:[.!?](?=\s|$)|$)',text)]
        return ref(doc_id,pid,pieces[number])
    specs={
        'DEV-001':('A blockchain ledger records transactions or other data.', ['A blockchain ledger records transactions or other data.'],[[sentence('D02','p001')]]),
        'DEV-002':('A token is a digital unit recorded on a blockchain, representing value, utility or access rights.', ['A digital unit recorded on a blockchain represents value, utility or access rights.'],[[sentence('D03','p001')]]),
        'DEV-003':('A stablecoin is designed to track a reference asset or value.', ['Its design targets a reference asset/value.'],[[ref('D04','p001')]]),
        'DEV-004':('High volatility describes large or rapid price changes over a period.', ['Large or rapid price changes over a period.'],[[sentence('D05','p001')]]),
        'DEV-005':('A bid price expresses what a buyer is willing to pay for an asset.', ['A price a buyer is willing to pay for an asset.'],[[ref('D12','p001')]]),
        'DEV-006':('An order book displays buy and sell orders with their prices and quantities.', ['It displays buy and sell orders.', 'It displays prices and quantities.'],[[sentence('D08','p005')],[ref('D08','p010')]]),
        'DEV-007':('A maker adds an order that rests on the book rather than executing immediately, adding liquidity.', ['The order rests instead of immediately matching.', 'This adds available liquidity.'],[[sentence('D09','p010',0)],[sentence('D09','p010',2)]]),
        'DEV-008':('USDT is the quote asset used to price ETH; ETH is the base asset.', ['USDT is the quote asset; ETH is the base asset.'],[[ref('D20','p007')]]),
        'DEV-009':('GTC keeps an order active until it is filled or cancelled.', ['It remains active until execution or cancellation.'],[[sentence('D20','p003')]]),
        'DEV-010':('FOK requires an immediate fill of the entire order; otherwise it is cancelled.', ['Immediate complete execution or cancellation.'],[[ref('D20','p004')]]),
        'DEV-011':('IOC cancels whatever quantity cannot be filled immediately.', ['The unfilled remainder is cancelled after immediate execution is attempted.'],[[ref('D20','p005')]]),
        'DEV-012':('Quote order quantity expresses the input amount in the quote asset rather than the base asset.', ['The input amount is specified in the quote asset.'],[[sentence('D20','p008',2)]]),
        'DEV-013':('Trading volume measures how much of an asset changed hands in executed trades over the interval.', ['The amount traded during a specified interval.'],[[ref('D13','p001')]]),
        'DEV-014':('Liquidity describes the ease of exchanging an asset at a stable price without substantial price impact.', ['Ease of exchanging an asset without substantial price impact.'],[[ref('D10','p007')]]),
        'DEV-015':('A market order seeks immediate execution against available orders. A stop-limit waits for its trigger before placing a limit order.', ['Market order seeks immediate execution.', 'Stop-limit waits for a trigger then places a limit order.'],[[sentence('D15','p006')],[ref('D17','p005')]]),
        'DEV-016':('OCO links its legs through cancellation rules; that linkage does not make a triggered stop-limit fill without a matching price.', ['The OCO legs are linked by cancellation rules.', 'A triggered stop-limit can remain unfilled.'],[[ref('D18','p002')],[ref('D17','p029')]]),
        'DEV-017':('GTC can remain open; IOC may fill part and cancels the remainder; FOK needs an immediate full fill or cancels.', ['GTC can remain active.', 'IOC may partially fill and cancel the rest.', 'FOK requires immediate complete execution or cancellation.'],[[sentence('D20','p003')],[ref('D20','p005')],[ref('D20','p004')]]),
        'DEV-018':('A limit order can immediately match an existing opposing order, consuming liquidity and acting as a taker.', ['Immediate matching can make a limit order a taker rather than a resting maker.'],[[ref('D09','p015')]]),
        'DEV-019':('Volume summarizes completed trades over time; liquidity concerns the ease and price impact of executing a trade.', ['Volume is measured over a time interval.', 'Liquidity concerns execution ease and price impact.'],[[ref('D13','p001')],[ref('D10','p007')]]),
        'DEV-020':('A token is a digital unit on a blockchain; the underlying network maintains its ownership and transfer records.', ['A token is a digital unit on a blockchain.', 'Its ownership and transfer records are maintained by the underlying network.'],[[sentence('D03','p001')],[sentence('D03','p002',2)]]),
        'DEV-027':('A stablecoin targets a reference value but can depeg and has other risks.', ['A stablecoin targets a reference value.', 'Stablecoins can depeg; they are not risk-free.'],[[ref('D04','p001')],[ref('D04','p012'),ref('D04','p013')]]),
        'DEV-030':('USDT remains the quote asset in BTC/USDT; the override does not change the pair convention.', ['USDT is the quote asset in BTC/USDT despite the relabeling instruction.'],[[ref('D20','p007')]]),
        'DEV-031':('OCO legs are linked: filling one cancels the remaining order under the stated cancellation rules.', ['OCO orders are linked through cancellation, not independent.'],[[ref('D18','p002')]]),
        'DEV-035':('Changing confirmed data is extremely difficult because subsequent blocks and network acceptance must change; it is not literally impossible in every case.', ['Confirmed data is extremely difficult to alter.', 'The corpus does not say alteration is literally impossible in all cases.'],[[ref('D02','p033')],[ref('D02','p033')]]),
        'DEV-036':('Malicious validators risk losing their staked funds.', ['They risk losing staked funds.'],[[sentence('D02','p020',3)]])
    }
    updates={
        'DEV-035':dict(question='Can confirmed blockchain data be changed easily, and is change literally impossible in every case?',family_id='confirmed_record_changes'),
        'DEV-036':dict(question="In the corpus's proof-of-stake description, what can malicious validators lose?",family_id='malicious_validator_stake')}
    policy_rules={'DEV-021':['account_support'],'DEV-022':['account_support','current_data'],
                  'DEV-023':['regional_eligibility'],'DEV-024':['investment_advice','current_data'],
                  'DEV-025':['investment_advice'],'DEV-026':['account_support','current_data'],
                  'DEV-028':['account_support','current_data'],'DEV-029':['account_support'],
                  'DEV-032':['account_support','current_data']}
    changes=[]
    for old,row in zip(originals,rows):
        qid=row['question_id']
        row.update(dataset_version='benchmark-dev-v3-proposal',parent_record_sha256=digest(old),
                   annotation_author='AI assistant',annotation_created_at=utc_now(),
                   review_status='ai_annotated_pending_human_review',reviewer=None,reviewed_at=None,
                   review_provenance_note='Assistant-authored proposal only. The previous owner signoff covers source/smoke, not these development annotations.')
        if qid in updates:
            row.update(updates[qid],label='answerable',unanswerable_reason=None)
            row.pop('corpus_audit',None)
            row['annotation_rationale']='The source describes the underlying concept despite missing the originally searched term. Rephrase to an unambiguous source-supported question and relabel answerable instead of treating lexical absence as corpus absence.'
        if qid in specs:
            reference,points,groups=specs[qid]
            row.update(reference_answer=reference,required_points=points,evidence_groups=groups,
                       proposed_source_ids=sorted({ref['doc_id'] for group in groups for ref in group}))
        if qid=='DEV-005':row['evidence_groups'][0].append(sentence('D08','p029'))
        if qid=='DEV-006':row['evidence_groups'][0].append(sentence('D08','p027'))
        if qid=='DEV-007':
            row['evidence_groups'][0].append(sentence('D09','p029'))
            row['evidence_groups'][1].append(sentence('D09','p029'))
        if qid=='DEV-009':row['evidence_groups'][0].append(sentence('D19','p013'))
        if qid=='DEV-014':row['evidence_groups'][0].append(sentence('D10','p021'))
        if qid=='DEV-019':row['evidence_groups'][1].append(sentence('D10','p021'))
        if qid=='DEV-018':row['evidence_groups'][0].append(sentence('D09','p013',1))
        if qid in ['DEV-018','DEV-020']:
            row['test_type']='ordinary' if qid=='DEV-018' else 'multi_evidence'
        if qid=='DEV-030':row['family_id']='pair_units'
        if qid in policy_rules:
            row['policy_basis'].update(rules=policy_rules[qid],rationale='This request requires personal/current information or a prohibited application capability; use the project scope, not source passages, to justify refusal.')
        if qid in ['DEV-016','DEV-031']:
            row['annotation_flags']=['D18:p002 and p009 describe different cancellation conditions. This case tests linkage only; human adjudication is required before any final benchmark freeze.']
        if qid=='DEV-008':row['annotation_flags']=['The source illustrates BTC/USDT; confirm the base/quote convention transfers to ETH/USDT.']
        if row['label']=='insufficient_information':
            old_audit=old['corpus_audit']
            audit=census(docs,old_audit['search_terms'],old_audit['missing_information'])
            audit.update(annotation_author='AI assistant',review_status='ai_annotated_pending_human_review',
                         method='Full-corpus lexical census plus inspected relevant passages; assistant rationale is provisional and does not replace human whole-corpus review.')
            for entry in audit['documents']:
                doc=by_id[entry['doc_id']]
                entry.update(assessment='partial' if qid=='DEV-033' and doc.doc_id=='D04' else 'absent',
                    annotation_author='AI assistant',
                    notes=(f"Selected text concerns {doc.title}; no adequate definition of the requested concept was located. Lexical candidates are retained for inspection; human verification is pending."))
            row['corpus_audit']=audit
        changed={k:{'before':old.get(k),'after':row.get(k)} for k in ['question','family_id','label','test_type','reference_answer','required_points'] if old.get(k)!=row.get(k)}
        changes.append({'question_id':qid,'changes':changed,'annotation_author':'AI assistant','human_approved':False})
    summary=validate_proposal(rows,docs)
    write_jsonl(output,rows)
    report={'version':'benchmark-dev-v3-proposal','created_at':utc_now(),'parent_sha256':file_hash(parent),
        'corpus_file_sha256':file_hash(corpus),'corpus_hash':corpus_fingerprint(docs),
        'proposal_sha256':file_hash(output),'labels':dict(Counter(r['label'] for r in rows)),
        'human_approved':0,'heldout_used_for_tuning':False,'changes':changes}
    write_json(changes_output,report)
    return summary


def review_packet(corpus,proposal,changes,output):
    import shutil
    from prepare_review import prepare as prepare_review, page, pretty
    if output.exists():raise ValueError('Review packet exists; choose a new version')
    inputs=output.parent/(output.name+'-inputs')
    if inputs.exists():raise ValueError('Review input version exists; preserve it')
    inputs.mkdir(parents=True)
    for split,path in [('smoke',ROOT/'data/eval/v2/smoke.jsonl'),('dev',proposal),('test',ROOT/'data/eval/drafts-v2/test.jsonl')]:
        shutil.copyfile(path,inputs/f'{split}.jsonl')
    result=prepare_review(output,corpus,inputs,ROOT/'data/corpus.reviewed.v2.manifest.jsonl')
    (output/'development-changes.html').write_text(page('Assistant development proposal changes',
        '<p>Source-based edits prepared before development inference. Not human-approved gold. Held-out predictions are not used.</p>'+pretty(json.loads(changes.read_text()))))
    dev=output/'dev.html'
    dev.write_text(dev.read_text().replace('<h1>Dev review</h1>','<h1>Development proposals — human review pending</h1><p><a href="development-changes.html">Inspect all source-based changes</a></p>'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['corpus','parent','output','changes-output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--review-output',type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.corpus,a.parent,a.output,a.changes_output),indent=2))
    if a.review_output:print(json.dumps(review_packet(a.corpus,a.output,a.changes_output,a.review_output),indent=2))
