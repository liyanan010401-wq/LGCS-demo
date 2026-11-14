"""Generate geometric retinal-like phantoms; no real patients or disease evidence."""
from pathlib import Path
import hashlib,json
import numpy as np


def generate(output=None,seed=20261030):
    output=Path(output) if output else Path(__file__).parent
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed);size=32
    yy,xx=np.mgrid[:size,:size];radius=14.;fov=(xx-15.5)**2+(yy-15.5)**2<=radius**2
    def picture(grade,peripheral):
        base=.1+.08*(1-((xx-15.5)**2+(yy-15.5)**2)/(radius**2))
        base+=rng.normal(0,.025,(size,size))
        for k in range(grade):
            angle=rng.uniform(0,2*np.pi)
            rad=rng.uniform(8,11) if peripheral else rng.uniform(2,7)
            cx=15.5+rad*np.cos(angle);cy=15.5+rad*np.sin(angle)
            sigma=rng.uniform(.7,1.7)
            base+=.95*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma**2))
        rgb=np.stack([base*rng.uniform(.93,1.05) for _ in range(3)])
        return np.clip(rgb*fov,0,1).astype(np.float32)
    data={}
    for split,count in [('source',96),('pool',80),('query',24)]:
        images=[];labels=[];grades=[]
        for i in range(count):
            if split=='source':grade=i%5;peripheral=bool(rng.integers(2))
            else:
                positive=(i%4)<2;peripheral=bool(i%2)
                grade=int(rng.integers(3,5) if positive else rng.integers(0,3))
            images.append(picture(grade,peripheral));grades.append(grade)
            labels.append([int(grade>=3),int(peripheral)])
        data[split+'_images']=np.stack(images)
        data[split+'_fov']=np.repeat(fov[None],count,0)
        data[split+'_ids']=np.array([f'{split}-phantom-{i:03d}' for i in range(count)])
        data[split+'_grades']=np.array(grades,dtype=np.int64)
        data[split+'_labels']=np.array(labels,dtype=np.int64)
    np.savez_compressed(output/'sample.npz',**data)
    meta={'kind':'synthetic smoke-test data, not EyePACS/RFMiD','seed':seed,'generator':'data/generate_data.py','license':'CC0-1.0 for these generated arrays only','units':{'images':'RGB in [0,1]','grid':'32x32 pixels','grades':'synthetic ordinal 0..4','labels':'two independent synthetic binary attributes, not clinical diagnoses'},'splits':{'source':96,'pool':80,'query':24},'all_identifiers_unique':True,'sha256':hashlib.sha256((output/'sample.npz').read_bytes()).hexdigest()}
    (output/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return data

if __name__=='__main__':generate()
