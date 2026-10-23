"""Pinned, optional local judge runtime; no answer generation or paid APIs."""
from __future__ import annotations
from dataclasses import replace
import json
import time
from .ai_judge import PROMPT,SCHEMAS,validate_result
from .io import digest,utc_now
from .local_provider import generation_runtime,build_prefix_enforcer
from .schemas import Config


class BudgetStop(RuntimeError):pass


def validate_config(config):
    expected={'model','model_revision','max_input_tokens','max_output_tokens','window_tokens','seed','max_gpu_seconds','pilot_min_pass'}
    if set(config)!=expected:raise ValueError('Unexpected judge configuration fields')
    if config['model']!='microsoft/Phi-4-mini-instruct' or config['model_revision']!='cfbefacb99257ffa30c83adab238a50856ac3083':
        raise ValueError('This experiment requires the preregistered pinned Phi judge')
    limits={'max_input_tokens':6144,'max_output_tokens':1024,'window_tokens':3000,'max_gpu_seconds':10800}
    for field,maximum in limits.items():
        if type(config[field]) is not int or not 1<=config[field]<=maximum:raise ValueError('Invalid judge limit: '+field)
    if config['seed']!=42 or type(config['seed']) is not int or config['pilot_min_pass']!=10:
        raise ValueError('Preregistered seed/pilot gate changed')


class LocalJudge:
    def __init__(self,config,output):
        validate_config(config)
        self.config=config;self.output=output;self.started=time.monotonic();self.cache={};self.calls=0
        self.runtime={};self.tokenizer=None
        self.adapter_config=replace(Config(),model=config['model'],model_revision=config['model_revision'])

    def load(self):
        self.torch,self.tokenizer,self.model,self.runtime=generation_runtime(self.config['model'],self.config['model_revision'])
        return self.runtime

    def count(self,value):
        if self.tokenizer is None:self.load()
        return len(self.tokenizer.encode(json.dumps(value,ensure_ascii=False),add_special_tokens=False))

    def ask(self,kind,payload):
        if time.monotonic()-self.started>=self.config['max_gpu_seconds']:raise BudgetStop('GPU wall-clock budget exhausted')
        key=digest({'config':self.config,'prompt':PROMPT,'schema':SCHEMAS[kind],'kind':kind,'payload':payload})
        if key in self.cache:return self.cache[key]
        record={'task_hash':key,'kind':kind,'grader_type':'llm','created_at':utc_now(),
                'model':self.config['model'],'revision':self.config['model_revision'],'result':None,'error':None,
                'raw_output':'','input_tokens':0,'output_tokens':0}
        started=time.perf_counter()
        try:
            if self.tokenizer is None:self.load()
            schema=json.dumps(SCHEMAS[kind],separators=(',',':'))
            messages=[{'role':'system','content':PROMPT+'\nTask: '+kind+'\nJSON schema: '+schema},
                      {'role':'user','content':json.dumps(payload,ensure_ascii=False)}]
            inputs=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                                    return_dict=True,return_tensors='pt')
            count=inputs['input_ids'].shape[-1];record['input_tokens']=count
            if count>self.config['max_input_tokens']:raise ValueError('Judge prompt exceeds input limit; evidence was NOT truncated')
            enforcer=build_prefix_enforcer(self.adapter_config,SCHEMAS[kind])
            inputs=inputs.to('cuda:0');self.torch.manual_seed(self.config['seed']);self.torch.cuda.synchronize()
            with self.torch.inference_mode():
                output=self.model.generate(**inputs,do_sample=False,max_new_tokens=self.config['max_output_tokens'],
                    pad_token_id=self.tokenizer.eos_token_id,prefix_allowed_tokens_fn=enforcer,logits_to_keep=1)
            self.torch.cuda.synchronize();new=output[0][count:];record['output_tokens']=len(new)
            record['raw_output']=self.tokenizer.decode(new,skip_special_tokens=True)
            eos=self.model.generation_config.eos_token_id
            endings={eos} if isinstance(eos,int) else set(eos or [])
            endings.add(self.tokenizer.eos_token_id)
            if len(new)>=self.config['max_output_tokens'] and int(new[-1]) not in endings:
                raise ValueError('Output limit reached; no JSON repair')
            record['result']=validate_result(kind,json.loads(record['raw_output']),payload)
        except Exception as exc:
            record['error']=type(exc).__name__+': '+str(exc)[:500]
        record['latency_ms']=(time.perf_counter()-started)*1000
        self.calls+=1;self.cache[key]=record
        with self.output.open('a') as handle:handle.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(f'judge task {self.calls}: {kind}, valid={record["error"] is None}',flush=True)
        return record
