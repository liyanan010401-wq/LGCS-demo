from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import torch
from torch.nn import functional as F

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
from src.core import TinySourceGraph,SourceInterface,BalancedSelector,ContextPredictor,average_precision
from data.generate_data import generate


def digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(config,output):
    predictor=ContextPredictor()
    if config.get('predictor')!=predictor.backend_name:
        raise ValueError('This entry point implements '+predictor.backend_name+'; integrate an external estimator explicitly before changing the backend name')
    torch.set_num_threads(1);torch.manual_seed(config['seed']);np.random.seed(config['seed'])
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=True)
    data_path=BASE/config['data']
    if not data_path.exists():generate(data_path.parent,config['seed'])
    with np.load(data_path,allow_pickle=False) as archive:data={k:archive[k].copy() for k in archive.files}
    ids=[set(data[s+'_ids'].tolist()) for s in ('source','pool','query')]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Partition identifier overlap')
    model=TinySourceGraph(config['max_nodes'],config['saliency_threshold'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    x=torch.from_numpy(data['source_images']);m=torch.from_numpy(data['source_fov']);y=torch.from_numpy(data['source_grades'])
    history=[]
    for epoch in range(config['source_epochs']):
        order=torch.randperm(len(x));losses=[]
        for indices in order.split(config['batch_size']):
            optimizer.zero_grad();logits,_,_,reg,_,_=model(x[indices],m[indices]);loss=F.cross_entropy(logits,y[indices])+reg
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite source training loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    model.freeze();frozen_hash=digest(model)
    features={};records={}
    with torch.no_grad():
        for split in ('source','pool','query'):
            _,g,_,_,s,record=model(torch.from_numpy(data[split+'_images']),torch.from_numpy(data[split+'_fov']))
            features[split]=(g.numpy(),s);records[split]=record
    interface=SourceInterface.fit(*features['source'],data['source_ids'])
    tables={split:interface.transform(*value) for split,value in features.items()}
    standardized={split:interface.standardize_distance(value) for split,value in tables.items()}
    selector=BalancedSelector(config['alpha'],config['beta'])
    predictions={rule:np.zeros((len(data['query_ids']),2)) for rule in ('graph','random','cosine','full')}
    ledger=[];endpoint_deltas=[]
    # Only the evaluator constructs libraries with pool truth. The predictor and
    # selector get one revealed disease column, never query labels or other labels.
    for disease in range(2):
        known=data['pool_labels'][:,disease];rng=np.random.default_rng(config['seed']+disease)
        library=np.concatenate([rng.permutation(np.flatnonzero(known==c))[:config['library_per_class']] for c in (0,1)])
        labels=known[library];library_ids=data['pool_ids'][library]
        disease_ledger={'disease':disease,'B':len(library),'library_ids':library_ids.tolist(),'query_contexts':[]}
        for qi,query in enumerate(standardized['query']):
            contexts={}
            for rule in ('graph','random','cosine'):
                selected=selector.select(query,standardized['pool'][library],labels,library_ids,config['context_per_class'],rule,config['seed']+qi)
                predictions[rule][qi,disease]=predictor.predict(tables['pool'][library[selected]],labels[selected],tables['query'][qi])[0]
                contexts[rule]=library_ids[selected].tolist()
            endpoint=[]
            for rule in ('graph','random','cosine'):
                selected=selector.select(query,standardized['pool'][library],labels,library_ids,config['library_per_class'],rule,config['seed']+qi)
                endpoint.append(predictor.predict(tables['pool'][library[selected]],labels[selected],tables['query'][qi])[0])
            endpoint_deltas.append(float(np.ptp(endpoint)));predictions['full'][qi,disease]=endpoint[0]
            disease_ledger['query_contexts'].append({'query_id':str(data['query_ids'][qi]),'k':2*config['context_per_class'],'selected':contexts})
        ledger.append(disease_ledger)
    if digest(model)!=frozen_hash:raise AssertionError('Target inference changed frozen encoder')
    if max(endpoint_deltas)>1e-12:raise AssertionError('Full-library endpoint mismatch')
    # Query ground truth first enters the evaluation below, after all predictions.
    evaluation={rule:[average_precision(data['query_labels'][:,c],value[:,c]) for c in range(2)] for rule,value in predictions.items()}
    report={'algorithm':'LGCS','status':'synthetic_smoke_test_only','predictor':config['predictor'],'source_training_loss':history,'source_rank':interface.rank,'feature_shape':list(tables['query'].shape),'predictions_shape':list(predictions['graph'].shape),'label_budget_per_disease':2*config['library_per_class'],'context_rows':2*config['context_per_class'],'source_model_unchanged_in_target':True,'full_library_max_probability_delta':max(endpoint_deltas),'synthetic_AP_by_attribute':evaluation,'empty_graphs':{k:sum(r['empty_graph'] for r in v) for k,v in records.items()},'elapsed_seconds':time.perf_counter()-start,'real_dataset_evaluation':False,'official_SIGraph_or_TabICL_reproduction':False,'versions':{'torch':torch.__version__,'numpy':np.__version__}}
    np.savez_compressed(output/'predictions.npz',**predictions,query_ids=data['query_ids'])
    np.savez_compressed(output/'source_interface.npz',mean=interface.mean,components=interface.components,distance_mean=interface.distance_mean,distance_scale=interface.distance_scale,source_ids=np.asarray(interface.source_ids))
    torch.save({'state_dict':model.state_dict(),'config':config,'scope':'tiny source-trained demo, not official SIGraph'},output/'tiny_source_model.pt')
    for name,value in [('metrics.json',report),('label_ledger.json',ledger),('graph_measurements.json',records),('config_used.json',config)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='LGCS synthetic CPU demonstration; not manuscript empirical evidence')
    parser.add_argument('--config',type=Path,default=BASE/'config.json')
    parser.add_argument('--output',type=Path,default=BASE/'outputs/demo')
    args=parser.parse_args();report=run(json.loads(args.config.read_text(encoding='utf-8')),args.output.resolve())
    print(json.dumps(report,ensure_ascii=False,indent=2))
