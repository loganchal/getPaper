#!/usr/bin/env python3
import itertools, math
import numpy as np
from scipy.spatial import ConvexHull


def rank_mod(A,p=1000003):
    if not A: return 0
    M=[[(int(x)%p) for x in row] for row in A]
    m=len(M); n=len(M[0]); r=0
    for c in range(n):
        q=next((i for i in range(r,m) if M[i][c]),None)
        if q is None: continue
        M[r],M[q]=M[q],M[r]
        z=pow(M[r][c],p-2,p)
        M[r]=[(x*z)%p for x in M[r]]
        for i in range(m):
            if i!=r and M[i][c]:
                z=M[i][c]
                M[i]=[(M[i][j]-z*M[r][j])%p for j in range(n)]
        r+=1
        if r==n: break
    return r


def analyze(stored_facets,need_all=False):
    T=-np.asarray(stored_facets,dtype=np.int64)
    m,d=T.shape; r=m-d
    assert np.array_equal(T[:d],np.eye(d,dtype=np.int64))
    hull=ConvexHull(T.astype(float))
    active_sets=sorted(set(tuple(sorted(map(int,I))) for I in hull.simplices))
    zero=[]; allc={}; verts=[]
    for I in active_sets:
        M=T[list(I)]
        inv=np.rint(np.linalg.inv(M.astype(float))).astype(np.int64)
        if not np.array_equal(M@inv,np.eye(d,dtype=np.int64)):
            raise RuntimeError(('nonunimodular',I,round(np.linalg.det(M))))
        x=inv@np.ones(d,dtype=np.int64)
        vals=T@x
        active=tuple(np.flatnonzero(vals==1).tolist())
        if active!=I or np.any(vals>1):
            raise RuntimeError(('hull mismatch',I,active,int(vals.max())))
        verts.append((I,x,vals,inv))
        for j in range(m):
            if j in I: continue
            slack=1-int(vals[j])
            c=np.zeros(r,dtype=np.int64)
            row=T[j]@inv
            for k,i in enumerate(I):
                if i>=d: c[i-d]+=row[k]
            if j>=d: c[j-d]-=1
            if np.any(c):
                q=next(int(z) for z in c if z)
                if q<0: c=-c
                ct=tuple(map(int,c)); K=slack-1
                if K==0: zero.append(ct)
                if need_all: allc[ct]=min(allc.get(ct,K),K)
    zr=rank_mod(zero)
    out={'m':m,'d':d,'r':r,'vertices':len(verts),'zero_constraints':len(set(zero)),'zero_rank':zr}
    if need_all: out['constraints']=sorted((c,K) for c,K in allc.items())
    return out


def kernel_basis_snf(zero,r):
    if not zero: return np.eye(r,dtype=np.int64)
    from sympy import ZZ
    from sympy.polys.matrices import DomainMatrix
    from sympy.polys.matrices.normalforms import smith_normal_decomp
    A=DomainMatrix([list(map(ZZ,row)) for row in zero],(len(zero),r),ZZ)
    D,S,V=smith_normal_decomp(A)
    dl=D.to_Matrix()
    rank=sum(1 for i in range(min(dl.rows,dl.cols)) if dl[i,i]!=0)
    Vm=V.to_Matrix()
    B=np.array([[int(Vm[i,j]) for j in range(rank,r)] for i in range(r)],dtype=np.int64)
    Z=np.asarray(zero,dtype=np.int64)
    if B.size and np.any(Z@B): raise RuntimeError('bad SNF kernel')
    return B


def exact_neat_search(stored_facets,enumerate_limit=5000000):
    info=analyze(stored_facets,need_all=True)
    r=info['r']; d=info['d']; m=info['m']; constraints=info['constraints']
    zero=[c for c,K in constraints if K==0]
    B=kernel_basis_snf(zero,r); k=B.shape[1]
    basic={q:info[q] for q in ('d','m','r','vertices','zero_constraints','zero_rank')}
    if k==0:
        return {**basic,'kernel_dimension':0,'status':'neat_trivial','type_lattice_points':1}
    best={}
    for c,K in constraints:
        q=tuple(map(int,np.asarray(c,dtype=np.int64)@B))
        if not any(q):
            if K<0: raise RuntimeError('infeasible zero inequality')
            continue
        if next(x for x in q if x)<0: q=tuple(-x for x in q)
        best[q]=min(best.get(q,K),K)
    CBK=sorted(best.items())
    C2=[]; K2=[]
    for c,K in CBK:
        C2.extend([c,tuple(-x for x in c)]); K2.extend([K,K])
    from scipy.optimize import linprog
    Cfloat=np.asarray(C2,dtype=float); Kfloat=np.asarray(K2,dtype=float)
    bounds=[]
    for i in range(k):
        obj=np.zeros(k); obj[i]=1
        lo=linprog(obj,A_ub=Cfloat,b_ub=Kfloat,bounds=[(None,None)]*k,method='highs')
        hi=linprog(-obj,A_ub=Cfloat,b_ub=Kfloat,bounds=[(None,None)]*k,method='highs')
        if not lo.success or not hi.success:
            return {**basic,'kernel_dimension':k,'kernel_basis':B.tolist(),'status':'unbounded_or_infeasible','linprog':(lo.message,hi.message)}
        bounds.append((math.ceil(lo.fun-1e-7),math.floor(-hi.fun+1e-7)))
    volume=math.prod(U-L+1 for L,U in bounds)
    base={**basic,'kernel_dimension':k,'kernel_basis':B.tolist(),'reduced_bounds':bounds,
          'reduced_box_volume':volume,'reduced_constraints':[(list(c),K) for c,K in CBK]}
    if volume>enumerate_limit: return {**base,'status':'box_too_large'}
    T=-np.asarray(stored_facets,dtype=np.int64)
    extra=T[d:]
    ternary=np.asarray(list(itertools.product((-1,0,1),repeat=d)),dtype=np.int16)
    centers=np.unique(ternary@extra.T,axis=0).astype(np.int64)
    feasible=0
    for z in itertools.product(*(range(L,U+1) for L,U in bounds)):
        if all(abs(sum(c[i]*z[i] for i in range(k)))<=K for c,K in CBK):
            feasible+=1
            bfree=B@np.asarray(z,dtype=np.int64)
            if not bool(np.any(np.all(np.abs(centers-bfree)<=1,axis=1))):
                b=np.concatenate([np.zeros(d,dtype=np.int64),bfree])
                for y in ternary.astype(np.int64):
                    if np.all(np.abs(T@y-b)<=1): raise RuntimeError(('coverage mismatch',y.tolist()))
                return {**base,'status':'non_neat','z':list(map(int,z)),'b':b.tolist(),
                        'type_lattice_points_checked':feasible,'outward_facets':T.tolist()}
    return {**base,'status':'neat_all_displacements','type_lattice_points':feasible}
