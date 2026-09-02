#!/usr/bin/env python3
import argparse, gzip, json, random, time, traceback
from collections import Counter
from pathlib import Path
from neat_fast import analyze, exact_neat_search


def reservoir_blocks(path,k,seed):
    rng=random.Random(seed); sample=[]; blocks=0; current=None
    with gzip.open(path,'rt') as stream:
        for line in stream:
            text=line.strip()
            if current is None:
                if not text: continue
                if text!='FACETS': raise ValueError((blocks,text))
                current=[]; blocks+=1
            elif not text:
                item=(blocks,current)
                if len(sample)<k: sample.append(item)
                else:
                    j=rng.randrange(blocks)
                    if j<k: sample[j]=item
                current=None
            else:
                vals=list(map(int,text.split()))
                if vals[0]!=1: raise ValueError((blocks,text))
                current.append(vals[1:])
    if current is not None: raise RuntimeError('archive ended inside block')
    return blocks,sample


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('archive'); ap.add_argument('--sample',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=0); ap.add_argument('--out',required=True)
    ap.add_argument('--enumerate-limit',type=int,default=2000000)
    args=ap.parse_args(); t0=time.time()
    total,sample=reservoir_blocks(args.archive,args.sample,args.seed)
    print('RESERVOIR',total,len(sample),round(time.time()-t0,2),flush=True)
    statuses=Counter(); deficiencies=Counter(); kernel_dims=Counter(); records=[]
    for q,(idx,F) in enumerate(sample,1):
        try:
            quick=analyze(F,need_all=False)
            deficiency=quick['r']-quick['zero_rank']; deficiencies[deficiency]+=1
            if deficiency==0:
                result={**quick,'kernel_dimension':0,'status':'neat_trivial','type_lattice_points':1}
            else:
                result=exact_neat_search(F,args.enumerate_limit)
            result['archive_id']=idx; result['stored_facets']=F
            statuses[result['status']]+=1
            kernel_dims[result.get('kernel_dimension',-1)]+=1
            if deficiency or result['status'] not in ('neat_trivial','neat_all_displacements'):
                records.append(result)
            if result['status']=='non_neat':
                print('NON_NEAT',idx,json.dumps(result,sort_keys=True),flush=True)
                break
        except Exception as e:
            result={'archive_id':idx,'status':'error','error':repr(e),'traceback':traceback.format_exc(),'stored_facets':F}
            statuses['error']+=1; records.append(result)
        if q%1000==0:
            print('PROGRESS',q,dict(statuses),'def',dict(deficiencies),'seconds',round(time.time()-t0,2),flush=True)
    def score(x):
        priority={'non_neat':10,'box_too_large':9,'unbounded_or_infeasible':8,'error':7}.get(x['status'],0)
        return (priority,x.get('kernel_dimension',-1),x.get('type_lattice_points',0),x.get('reduced_box_volume',0))
    records=sorted(records,key=score,reverse=True)[:200]
    out={'archive':args.archive,'total_blocks':total,'sample_requested':args.sample,'sample_processed':sum(statuses.values()),
         'seed':args.seed,'statuses':dict(statuses),'zero_rank_deficiency':dict(deficiencies),
         'kernel_dimensions':dict(kernel_dims),'seconds':time.time()-t0,'interesting':records}
    Path(args.out).write_text(json.dumps(out,indent=2))
    print('FINAL',json.dumps({k:v for k,v in out.items() if k!='interesting'},sort_keys=True),flush=True)

if __name__=='__main__': main()
