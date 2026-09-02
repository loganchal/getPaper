#!/usr/bin/env python3
import argparse, gzip, json, math, random, time
from itertools import combinations, product
from pathlib import Path
import numpy as np


def make_reps(d):
    out=[]
    for x in product((-1,0,1), repeat=d):
        if not any(x):
            continue
        if next(q for q in x if q)>0:
            out.append(x)
    return out


class BasisEngine:
    def __init__(self,d):
        self.d=d
        self.R=make_reps(d)
        self.N=len(self.R)
        self.arr=np.asarray(self.R,dtype=np.int16)
        self.order=sorted(range(self.N),key=lambda i:(sum(q!=0 for q in self.R[i]),self.R[i]))
        subs=[list(combinations(range(d),k)) for k in range(d+1)]
        subidx=[{s:i for i,s in enumerate(ss)} for ss in subs]
        self.rules=[]
        for k in range(d):
            level=[]
            for I in subs[k+1]:
                level.append(tuple((subidx[k][I[:r]+I[r+1:]],c,-1 if ((r+k)&1) else 1)
                                   for r,c in enumerate(I)))
            self.rules.append(tuple(level))

    def extend(self,w,v,k):
        return tuple(sum(s*v[c]*w[j] for j,c,s in terms) for terms in self.rules[k])

    @staticmethod
    def primitive(w):
        g=0
        for z in w:
            if z:
                g=math.gcd(g,abs(z))
                if g==1:
                    return True
        return False

    @staticmethod
    def canonical(w):
        for z in w:
            if z<0:
                return tuple(-q for q in w)
            if z>0:
                return w
        return w

    def find_basis(self,mask,order=None):
        order=self.order if order is None else order
        w=(1,)
        basis_mask=0
        for k in range(self.d):
            for i in order:
                if (mask>>i)&1:
                    new=self.extend(w,self.R[i],k)
                    if self.primitive(new):
                        w=self.canonical(new)
                        basis_mask |= 1<<i
                        break
            else:
                return None
        if w!=(1,) or basis_mask.bit_count()!=self.d:
            raise AssertionError((w,basis_mask.bit_count()))
        return basis_mask

    def random_basis(self,mask,seed,tries=500):
        ids=[i for i in self.order if (mask>>i)&1]
        rng=random.Random(seed)
        for _ in range(tries):
            rng.shuffle(ids)
            b=self.find_basis(mask,ids)
            if b is not None:
                return b
        return None

    def normal_mask(self,a):
        ok=np.abs(self.arr@np.asarray(a,dtype=np.int16))<=1
        return int.from_bytes(np.packbits(ok,bitorder='little').tobytes(),'little')

    def vectors(self,b):
        return [list(self.R[i]) for i in range(self.N) if (b>>i)&1]

    def determinant(self,b):
        ids=[i for i in range(self.N) if (b>>i)&1]
        M=[[self.R[c][r] for c in ids] for r in range(self.d)]
        sign=1
        previous=1
        for k in range(self.d-1):
            if M[k][k]==0:
                row=next((r for r in range(k+1,self.d) if M[r][k]),None)
                if row is None:
                    return 0
                M[k],M[row]=M[row],M[k]
                sign=-sign
            pivot=M[k][k]
            for i in range(k+1,self.d):
                for j in range(k+1,self.d):
                    M[i][j]=(M[i][j]*pivot-M[i][k]*M[k][j])//previous
            previous=pivot
        return sign*M[-1][-1]


def canonical_normal(a):
    for z in a:
        if z<0:
            return tuple(-q for q in a)
        if z>0:
            return a
    return a


def scan(path,d,out_path):
    engine=BasisEngine(d)
    all_mask=(1<<engine.N)-1
    index={v:i for i,v in enumerate(engine.R)}
    coordinate_basis=0
    for i in range(d):
        e=tuple(1 if j==i else 0 for j in range(d))
        coordinate_basis |= 1<<index[e]

    # [basis mask, number of later polytopes covered]
    catalogue=[[coordinate_basis,0]]
    normal_masks={}
    unresolved=[]
    smallest=[]
    blocks=rows=checks=created=0
    in_block=False
    mask=all_mask
    facets=[]
    t0=time.time()

    with gzip.open(path,'rt') as stream:
        for line in stream:
            text=line.strip()
            if not in_block:
                if not text:
                    continue
                if text!='FACETS':
                    raise ValueError((blocks,text))
                blocks+=1
                mask=all_mask
                facets=[]
                in_block=True
            elif not text:
                n=mask.bit_count()
                if len(smallest)<25:
                    smallest.append((n,blocks,[list(a) for a in facets]))
                    smallest.sort(key=lambda q:(q[0],q[1]))
                elif n<smallest[-1][0]:
                    smallest[-1]=(n,blocks,[list(a) for a in facets])
                    smallest.sort(key=lambda q:(q[0],q[1]))

                for entry in catalogue:
                    checks+=1
                    b=entry[0]
                    if mask&b==b:
                        entry[1]+=1
                        break
                else:
                    b=engine.find_basis(mask)
                    if b is None:
                        b=engine.random_basis(mask,blocks,tries=1000)
                    if b is None:
                        unresolved.append({
                            'id':blocks,
                            'pair_representatives':n,
                            'facets':[list(a) for a in facets]
                        })
                        print('UNRESOLVED',blocks,n,flush=True)
                    else:
                        det=engine.determinant(b)
                        if abs(det)!=1:
                            raise AssertionError((blocks,det))
                        catalogue.append([b,1])
                        created+=1

                in_block=False
                if blocks%10000==0:
                    catalogue.sort(key=lambda q:q[1],reverse=True)
                    partial={
                        'dimension':d,'file':str(path),'blocks':blocks,'rows':rows,
                        'candidate_sign_representatives':engine.N,
                        'normal_masks':len(normal_masks),'catalogue_size':len(catalogue),
                        'created':created,'subset_checks':checks,
                        'unresolved':unresolved,
                        'smallest':[{'pair_representatives':n,'id':i,'facets':f} for n,i,f in smallest],
                        'seconds':time.time()-t0
                    }
                    Path(out_path).write_text(json.dumps(partial,indent=2))
                    print('PROGRESS',blocks,'catalogue',len(catalogue),'unresolved',len(unresolved),
                          'checks',checks,'normals',len(normal_masks),'seconds',round(time.time()-t0,2),flush=True)
            else:
                values=text.split()
                if len(values)!=d+1 or values[0]!='1':
                    raise ValueError((blocks,text))
                a=tuple(map(int,values[1:]))
                facets.append(a)
                rows+=1
                ca=canonical_normal(a)
                allowed=normal_masks.get(ca)
                if allowed is None:
                    allowed=engine.normal_mask(ca)
                    normal_masks[ca]=allowed
                mask &= allowed

    if in_block:
        raise RuntimeError('archive ended inside a block')

    certificates=[]
    for b,hits in catalogue:
        det=engine.determinant(b)
        if abs(det)!=1:
            raise AssertionError((det,b))
        certificates.append({'hits':hits,'determinant':det,'vectors':engine.vectors(b)})

    result={
        'dimension':d,'file':str(path),'blocks':blocks,'rows':rows,
        'candidate_sign_representatives':engine.N,
        'normal_masks':len(normal_masks),'catalogue_size':len(catalogue),
        'created':created,'subset_checks':checks,
        'unresolved':unresolved,
        'smallest':[{'pair_representatives':n,'id':i,'facets':f} for n,i,f in smallest],
        'seconds':time.time()-t0,
        'catalogue':certificates
    }
    Path(out_path).write_text(json.dumps(result,indent=2))
    print('FINAL',json.dumps({k:v for k,v in result.items() if k not in ('catalogue','smallest','unresolved')},sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive')
    parser.add_argument('--dimension',type=int,required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    scan(Path(args.archive),args.dimension,Path(args.out))


if __name__=='__main__':
    main()
