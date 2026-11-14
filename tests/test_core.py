from pathlib import Path
import sys,unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import spatial_statistics,SourceInterface,BalancedSelector,ContextPredictor,TinySourceGraph,average_precision
torch.set_num_threads(1)

class InterfaceTests(unittest.TestCase):
    def test_density_and_salient_fraction_are_distinct(self):
        coords=np.array([[0,0],[1,0],[2,2]])
        assignment=np.array([[.5,.5],[.9,.1],[.1,.9]])
        adjacency=np.ones((3,3),bool)
        s,r=spatial_statistics(coords,assignment,adjacency,np.ones((3,3),bool))
        self.assertEqual(r['areas'],[2,1]);self.assertAlmostEqual(s[0],2/9);self.assertAlmostEqual(s[1],3/9)
        self.assertEqual(r['edges'],[(0,1)]);self.assertAlmostEqual(s[-1],1/6)

    def test_empty_is_not_invalid_fov(self):
        s,r=spatial_statistics([],np.empty((0,6)),np.empty((0,0)),np.ones((3,3),bool))
        np.testing.assert_array_equal(s,np.zeros(8));self.assertTrue(r['empty_graph'])
        with self.assertRaises(ValueError):spatial_statistics([],[],[],np.zeros((3,3),bool))

    def test_source_fit_rejects_target_and_low_rank(self):
        with self.assertRaises(ValueError):SourceInterface.fit(np.ones((40,64)),np.zeros((40,8)),list(range(40)))
        with self.assertRaises(ValueError):SourceInterface.fit(np.ones((40,64)),np.zeros((40,8)),list(range(40)),partition='target')

    def test_context_budget_and_endpoint(self):
        rng=np.random.default_rng(9);library=rng.normal(size=(40,40));y=np.tile([0,1],20);ids=np.array([f'i{i:02}' for i in range(40)]);q=rng.normal(size=40)
        selector=BalancedSelector();predictor=ContextPredictor();values=[]
        for rule in ('random','cosine','graph'):
            ix=selector.select(q,library,y,ids,5,rule);self.assertEqual(len(ix),10);self.assertEqual(y[ix].sum(),5)
            full=selector.select(q,library,y,ids,20,rule);self.assertEqual(ids[full].tolist(),sorted(ids.tolist()))
            values.append(predictor.predict(library[full],y[full],q)[0])
        self.assertEqual(np.ptp(values),0)

    def test_beta_zero_is_class_balanced_nearest(self):
        x=np.zeros((6,40));x[:,0]=[6,1,2,5,3,4];ids=np.array(['a','b','c','d','e','f']);y=np.array([0,0,0,1,1,1])
        ix=BalancedSelector(beta=0).select(np.zeros(40),x,y,ids,1)
        self.assertEqual(set(ids[ix]),{'b','e'})

    def test_query_batch_does_not_fit_preprocessing(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(10,40));q=rng.normal(size=(4,40));y=np.tile([0,1],5);p=ContextPredictor()
        np.testing.assert_allclose(p.predict(x,y,q),np.concatenate([p.predict(x,y,row) for row in q]),atol=1e-14)
        combined=np.concatenate((q,np.full((1,40),1e8)))
        np.testing.assert_allclose(p.predict(x,y,q),p.predict(x,y,combined)[:4],atol=1e-14)

    def test_encoder_has_source_gradients_and_freezes(self):
        torch.manual_seed(3);model=TinySourceGraph(max_nodes=24)
        image=torch.rand(2,3,16,16);mask=torch.ones(2,16,16,dtype=torch.bool)
        logits,g,_,reg,s,_=model(image,mask);(logits.square().mean()+reg).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))
        self.assertEqual(g.shape,(2,64));self.assertEqual(s.shape,(2,8))
        model.freeze();self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_tied_scores_use_grouped_AP(self):
        self.assertAlmostEqual(average_precision([1,0],[.5,.5]),.5)
        self.assertIsNone(average_precision([0,0],[.2,.1]))

    def test_AP_all_positive_empty_and_shape_contract(self):
        self.assertEqual(average_precision([1,1],[.2,.9]),1.)
        self.assertIsNone(average_precision([],[]))
        with self.assertRaises(ValueError):average_precision([1,0],[.2])

    def test_unsupported_backend_is_rejected_before_loading_data(self):
        from run_demo import run
        with self.assertRaisesRegex(ValueError,'integrate an external estimator'):
            run({'predictor':'TabICL'},Path('never_created'))

if __name__=='__main__':unittest.main()
