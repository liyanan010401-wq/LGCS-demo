"""LGCS core: source fitting, hard spatial measurement, and label-budgeted contexts.

The small encoder and default kernel context learner are explicit demo substitutes
for SIGraph and TabICL v1. The 32+8 interface and selector follow the manuscript.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def spatial_statistics(coords, assignments, adjacency, fov):
    """Measure integer pixel areas after cumulative soft assignments.

    coords: selected analysis-grid pixel coordinates (x,y), not arbitrary patches.
    assignments: selected pixels x final clusters, nonnegative soft weights.
    Empty clusters are discarded. Edges are projected, undirected and deduplicated.
    """
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    fov = np.asarray(fov, dtype=bool)
    valid_yx = np.argwhere(fov)
    if len(valid_yx) < 2:
        raise ValueError('Invalid FOV: at least two valid pixels required')
    valid = valid_yx[:, ::-1].astype(np.float64)
    center = valid.mean(0)
    radius = np.linalg.norm(valid-center, axis=1).max()
    if radius <= 0:
        raise ValueError('Invalid zero-radius FOV')
    if len(coords) == 0:
        return np.zeros(8), {'empty_graph': True, 'edge_free': True, 'N': 0, 'P_s': 0, 'areas': [], 'edges': []}
    assignment = np.asarray(assignments, dtype=np.float64)
    adjacency = np.asarray(adjacency, dtype=bool)
    if assignment.shape[0] != len(coords) or adjacency.shape != (len(coords),len(coords)):
        raise ValueError('Graph/assignment dimensions disagree')
    if not np.isfinite(assignment).all() or (assignment < 0).any() or (assignment.sum(1) <= 0).any():
        raise ValueError('Invalid soft assignments')
    xy = coords.astype(int)
    if not np.allclose(xy, coords) or ((xy < 0).any()) or (xy[:,0] >= fov.shape[1]).any() or (xy[:,1] >= fov.shape[0]).any() or not fov[xy[:,1], xy[:,0]].all():
        raise ValueError('Initial nodes must be valid FOV analysis-grid pixels')
    hard = assignment.argmax(1)  # NumPy resolves ties at lowest stable index.
    labels = np.unique(hard)
    remap = {int(label): n for n,label in enumerate(labels)}
    compact = np.array([remap[int(v)] for v in hard])
    areas = np.bincount(compact, minlength=len(labels))
    centroids = np.stack([coords[compact == j].mean(0) for j in range(len(labels))])
    u = (centroids-center)/radius
    edges = sorted({tuple(sorted((int(compact[i]),int(compact[j])))) for i,j in np.argwhere(adjacency) if compact[i] != compact[j]})
    lengths = np.array([np.linalg.norm(u[i]-u[j]) for i,j in edges])
    radii = np.linalg.norm(u,axis=1)
    centered = u-u.mean(0)
    eig = np.linalg.eigvalsh(centered.T@centered/len(u))
    ratio = max(float(eig[0]),0)/(max(float(eig[-1]),0)+1e-8)
    gini = np.abs(areas[:,None]-areas[None,:]).sum()/(2*len(areas)*areas.sum())
    values = np.array([len(labels)/fov.sum(),len(coords)/fov.sum(),lengths.mean() if len(lengths) else 0,lengths.std() if len(lengths) else 0,radii.mean(),radii.std(),ratio,gini])
    return values, {'empty_graph': False,'edge_free': len(edges)==0,'N':len(labels),'P_s':len(coords),'areas':areas.tolist(),'edges':edges,'hard_assignments':compact.tolist(),'centroids':u.tolist()}


class TinySourceGraph(nn.Module):
    """Trainable source-only encoder; two soft pooling stages, not SIGraph weights."""
    def __init__(self, max_nodes=64, saliency_threshold=.58):
        super().__init__()
        self.max_nodes=max_nodes
        self.saliency_threshold=saliency_threshold
        self.cnn=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.GELU(),nn.Conv2d(16,32,3,padding=1),nn.GELU())
        self.node=nn.Linear(34,32)
        self.q=nn.Linear(32,32,bias=False)
        self.k=nn.Linear(32,32,bias=False)
        self.v=nn.Linear(32,32)
        self.pool1=nn.Linear(32,12)
        self.pool2=nn.Linear(32,6)
        self.lift=nn.Linear(32,64)
        self.global_lift=nn.Linear(32,64)
        self.classifier=nn.Linear(128,5)
        a=torch.arange(12)*2*math.pi/12
        b=torch.arange(6)*2*math.pi/6
        self.register_buffer('centers1',.65*torch.stack((a.cos(),a.sin()),1))
        self.register_buffer('centers2',.6*torch.stack((b.cos(),b.sin()),1))

    def _graph(self, image, feature, fov):
        points=torch.nonzero(fov,as_tuple=False)
        if len(points)<2:
            raise ValueError('Invalid FOV')
        xy_all=points[:,[1,0]].to(feature.dtype)
        center=xy_all.mean(0)
        radius=torch.linalg.vector_norm(xy_all-center,dim=1).max()
        if radius <= 0:
            raise ValueError('Invalid zero-radius FOV')
        saliency=image.detach().mean(0)
        selected=torch.nonzero(fov & (saliency>self.saliency_threshold),as_tuple=False)
        if len(selected)>self.max_nodes:
            ranks=torch.argsort(saliency[selected[:,0],selected[:,1]],descending=True,stable=True)[:self.max_nodes]
            selected=selected[ranks]
        if not len(selected):
            stats,record=spatial_statistics(np.empty((0,2)),np.empty((0,6)),np.empty((0,0)),fov.cpu().numpy())
            return feature.sum()*0+torch.zeros(64,device=feature.device),feature.sum()*0,stats,record
        coords=selected[:,[1,0]].to(feature.dtype)
        normalized=(coords-center)/radius
        h=F.gelu(self.node(torch.cat((feature[:,selected[:,0],selected[:,1]].T,normalized),1)))
        d=torch.cdist(normalized,normalized)
        neighbors=torch.argsort(d,dim=1,stable=True)[:,:min(5,len(d))]
        adjacency=torch.zeros_like(d,dtype=torch.bool).scatter_(1,neighbors,True)
        adjacency=adjacency|adjacency.T
        attention=(self.q(h)@self.k(h).T/math.sqrt(32)).masked_fill(~adjacency,-1e9).softmax(1)
        h=F.gelu(h+attention@self.v(h))
        s1=(self.pool1(h)-5*torch.cdist(normalized,self.centers1).square()).softmax(1)
        mass1=s1.sum(0).clamp_min(1e-8)
        c1=(s1.T@h)/mass1[:,None]
        pos1=(s1.T@normalized)/mass1[:,None]
        s2=(self.pool2(c1)-5*torch.cdist(pos1,self.centers2).square()).softmax(1)
        c2=(s2.T@F.gelu(self.lift(c1)))/s2.sum(0).clamp_min(1e-8)[:,None]
        a=adjacency.to(h.dtype)
        cut=-torch.trace(s1.T@a@s1)/((a.sum(1)[:,None]*s1.square()).sum().clamp_min(1e-8))
        gram=s1.T@s1
        orth=torch.linalg.matrix_norm(gram/torch.linalg.matrix_norm(gram).clamp_min(1e-8)-torch.eye(12,device=h.device)/math.sqrt(12))
        # Measurement is detached: hard areas never replace the differentiable readout.
        stats,record=spatial_statistics(coords.detach().cpu().numpy(),(s1@s2).detach().cpu().numpy(),adjacency.cpu().numpy(),fov.cpu().numpy())
        return c2.mean(0),.2*cut+.1*orth,stats,record

    def forward(self,images,fov):
        feature=self.cnn(images)
        graph=[];regularizers=[];stats=[];records=[]
        for x,h,m in zip(images,feature,fov):
            g,r,s,record=self._graph(x,h,m)
            graph.append(g);regularizers.append(r);stats.append(s);records.append(record)
        graph=torch.stack(graph)
        global_readout=F.gelu(self.global_lift(feature.mean((2,3))))
        logits=self.classifier(torch.cat((graph,global_readout),1))
        return logits,graph,global_readout,torch.stack(regularizers).mean(),np.stack(stats),records

    def freeze(self):
        self.eval()
        for p in self.parameters():p.requires_grad_(False)
        return self


@dataclass
class SourceInterface:
    mean: np.ndarray
    components: np.ndarray
    distance_mean: np.ndarray
    distance_scale: np.ndarray
    rank: int
    source_ids: list[str]

    @classmethod
    def fit(cls,graph_readouts,stats,source_ids,partition='source_train'):
        if partition!='source_train':raise ValueError('Projection/scales may only fit source_train')
        h=np.asarray(graph_readouts,dtype=np.float64);s=np.asarray(stats,dtype=np.float64)
        if h.shape[0]<33 or h.shape[1]<32:raise ValueError('32-component interface requires enough source rows and width')
        if not np.isfinite(h).all() or not np.isfinite(s).all():raise ValueError('Nonfinite source features')
        mean=h.mean(0);_,singular,vt=np.linalg.svd(h-mean,full_matrices=False)
        rank=int((singular>max(singular[0]*1e-10,1e-14)).sum())
        if rank<32:raise ValueError(f'Source numerical rank {rank} is below 32; revise before target evaluation')
        components=vt[:32].copy()
        for v in components:
            if v[np.argmax(abs(v))]<0:v*=-1
        table=np.concatenate(((h-mean)@components.T,s),1)
        scale=table.std(0);scale[scale==0]=1
        return cls(mean,components,table.mean(0),scale,rank,list(source_ids))

    def transform(self,graph_readouts,stats):
        h=np.asarray(graph_readouts,dtype=np.float64);s=np.asarray(stats,dtype=np.float64)
        if h.shape[-1]!=self.mean.size or s.shape[-1]!=8:raise ValueError('Interface width mismatch')
        row=np.concatenate(((h-self.mean)@self.components.T,s),axis=-1)
        if not np.isfinite(row).all():raise ValueError('Nonfinite target features')
        return row

    def standardize_distance(self,table):return (table-self.distance_mean)/self.distance_scale


class BalancedSelector:
    def __init__(self,alpha=.7,beta=.25):
        if not (0<=alpha<=1 and 0<=beta<=1):raise ValueError('Distance/coverage mixture must be in [0,1]')
        self.alpha=alpha;self.beta=beta

    def distance(self,a,b):
        diff=np.asarray(a)[:,None,:]-np.asarray(b)[None,:,:]
        return self.alpha*np.mean(diff[:,:,:32]**2,2)+(1-self.alpha)*np.mean(diff[:,:,32:]**2,2)

    def select(self,query,library,labels,ids,k_per_class,rule='graph',seed=0):
        x=np.asarray(library,dtype=float);y=np.asarray(labels);ids=np.asarray(ids)
        if x.ndim!=2 or x.shape[1]!=40 or len(ids)!=len(x) or len(y)!=len(x):raise ValueError('Invalid library contract')
        if len(set(ids.tolist()))!=len(ids):raise ValueError('Duplicate library identifiers')
        if not np.isin(y,[0,1]).all() or k_per_class<1:raise ValueError('Only revealed binary labels are accepted')
        if not np.isfinite(x).all() or not np.isfinite(query).all():raise ValueError('Nonfinite features')
        if rule not in ('graph','random','cosine'):raise ValueError('Unknown selection rule')
        rng=np.random.default_rng(seed);chosen=[]
        for label in (0,1):
            ix=np.flatnonzero(y==label);ix=ix[np.argsort(ids[ix],kind='stable')]
            if len(ix)<k_per_class:raise ValueError('Insufficient revealed class budget')
            if len(ix)==k_per_class:chosen.extend(ix);continue
            if rule=='random':chosen.extend(rng.choice(ix,k_per_class,replace=False));continue
            if rule=='cosine':
                norms=np.linalg.norm(x[ix],axis=1);qnorm=np.linalg.norm(query)
                distances=np.full(len(ix),2.)
                valid=(norms>0)&(qnorm>0)
                if valid.any():distances[valid]=1-(x[ix][valid]@query)/(norms[valid]*qnorm)
                chosen.extend(ix[np.argsort(distances,kind='stable')[:k_per_class]]);continue
            dq=self.distance(np.asarray(query)[None,:],x[ix])[0]
            dll=self.distance(x[ix],x[ix]);selected=[];remaining=list(range(len(ix)))
            for _ in range(k_per_class):
                values=[]
                for candidate in remaining:
                    trial=selected+[candidate]
                    objective=(1-self.beta)*dq[trial].mean()+self.beta*dll[:,trial].min(1).mean()
                    values.append(objective)
                best=remaining[int(np.argmin(values))]
                selected.append(best);remaining.remove(best)
            chosen.extend(ix[selected])
        chosen=np.asarray(chosen,dtype=int)
        return chosen[np.argsort(ids[chosen],kind='stable')]


class ContextPredictor:
    """Support-only preprocessing with an explicit replaceable classifier contract.

    Default is a deterministic RBF context demonstration, NOT a TabICL checkpoint.
    Pass estimator_factory returning official sklearn-style fit/predict_proba
    objects to integrate a locally provisioned, verified TabICL v1 checkpoint.
    """
    def __init__(self,estimator_factory=None,temperature=.8):
        self.estimator_factory=estimator_factory;self.temperature=temperature
    @property
    def backend_name(self):
        return 'deterministic_kernel_demo_not_TabICL' if self.estimator_factory is None else 'external_estimator_factory'
    def predict(self,support,labels,queries):
        support=np.asarray(support,dtype=float);queries=np.atleast_2d(queries);labels=np.asarray(labels)
        if not np.isfinite(support).all() or not np.isfinite(queries).all():raise ValueError('Nonfinite table')
        if set(labels.tolist())!={0,1}:raise ValueError('Both known classes are required')
        mean=support.mean(0);scale=support.std(0);scale[scale==0]=1
        xs=(support-mean)/scale;xq=(queries-mean)/scale
        if self.estimator_factory is not None:
            estimator=self.estimator_factory();estimator.fit(xs,labels)
            positive=np.flatnonzero(np.asarray(estimator.classes_)==1)
            if len(positive)!=1:raise ValueError('Estimator positive-class column missing')
            return estimator.predict_proba(xq)[:,positive[0]]
        logw=-np.mean((xq[:,None,:]-xs[None,:,:])**2,2)/(2*self.temperature**2)
        weights=np.exp(logw-logw.max(1,keepdims=True))
        return (weights@labels)/weights.sum(1)


def average_precision(labels,scores):
    """Step-weighted AP, grouped ties; empty/no positives undefined, all positive 1."""
    labels=np.asarray(labels);scores=np.asarray(scores)
    if labels.ndim!=1 or scores.shape!=labels.shape:raise ValueError('Evaluation arrays must be equal one-dimensional shapes')
    if not np.isin(labels,[0,1]).all() or not np.isfinite(scores).all():raise ValueError('Invalid evaluation arrays')
    if labels.sum()==0:return None
    order=np.argsort(-scores,kind='stable');y=labels[order];p=scores[order]
    ends=np.r_[np.flatnonzero(p[1:]!=p[:-1]),len(p)-1]
    tp=np.cumsum(y)[ends];recall=tp/y.sum();precision=tp/(ends+1)
    return float(np.sum(np.diff(np.r_[0.,recall])*precision))
