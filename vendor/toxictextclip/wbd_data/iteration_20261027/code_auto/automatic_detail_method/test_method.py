"""Contract checks for the method's substantive constraints, not its ASR."""
import types
import unittest
from collections import Counter

import torch

from .core import Config, ContextualDetails, choose_slots, mask_class, coefficient_for_round
from .native import additive_hook
from .pool import merge_pool, bank_candidates


class Contracts(unittest.TestCase):
    def test_identity_mask(self):
        self.assertEqual(mask_class('A dog, dogs and puppies <PERSON> on grass','dog'),['a','and','on','grass'])
        self.assertNotIn('television',mask_class('A television next to a couch','tv'))

    def test_contextual_deletions(self):
        calls = []
        table = {'white grass': [1.,0.,0.], 'grass':[0.,1.,0.], 'white':[0.,0.,1.], '':[0.,1.,1.]}
        def encode(texts):
            calls.extend(texts)
            return torch.nn.functional.normalize(torch.tensor([table[x] for x in texts]),dim=-1)
        center = torch.tensor([[0.,1.,0.],[0.,0.,1.]])
        engine = ContextualDetails(encode,center,['dog','car'],Config(eta=0,detail_k=2))
        out = engine.analyze('dog white grass','dog',torch.tensor([1.,0.,0.]))
        self.assertEqual(out['background'],'white grass')
        self.assertEqual(out['details'],['white'])
        self.assertTrue(out['grounding_pass'])
        self.assertNotIn('dog',out['detail_text'])
        self.assertIn('white grass',calls)
        self.assertIn('grass',calls)

    def test_no_force_deletion(self):
        def encode(texts):
            table={'white grass':[1.,0.,0.],'white':[0.,1.,0.],'grass':[0.,1.,0.],'':[0.,1.,0.]}
            return torch.tensor([table[x] for x in texts])
        engine = ContextualDetails(encode,torch.tensor([[0.,1.,0.],[0.,0.,1.]]),['dog','car'])
        self.assertFalse(engine.analyze('dog white grass','dog',torch.tensor([1.,0.,0.]))['greedy_deletions'])

    def test_gate_then_rank_and_no_one_per_class_cap(self):
        def row(caption,score,passed=True):
            return dict(caption=caption,path=caption+'.jpg',category='car',
                analysis=dict(grounding_pass=passed,background_class_cos=score))
        old = [row('old'+str(i),.1) for i in range(5)]
        new = [row('new'+str(i),.2+i*.01) for i in range(5)]+[row('ungrounded',.99,False)]
        chosen,decisions = choose_slots(old,new)
        self.assertEqual(len(decisions),5)
        self.assertNotIn('ungrounded',[x['caption'] for x in chosen])
        self.assertEqual(Counter(x['category'] for x in chosen),Counter(car=5))

    def test_empty_alternatives_are_not_certified(self):
        old = [dict(caption=str(i),category='car',analysis=dict(grounding_pass=False,background_class_cos=.1)) for i in range(5)]
        chosen,decisions = choose_slots(old,[])
        self.assertEqual(chosen,old)
        self.assertFalse(decisions)
        self.assertTrue(all(not x['analysis']['grounding_pass'] for x in chosen))

    def test_additive_feature_and_patch_preservation(self):
        author = types.SimpleNamespace(clip=types.SimpleNamespace(tokenize=lambda texts:torch.ones(len(texts),2,dtype=torch.long)))
        model = types.SimpleNamespace(clip_model=types.SimpleNamespace(encode_text=lambda token:torch.tensor([[2.,3.,4.]])),
            text_decoder=types.SimpleNamespace(linear_feature2eot=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])))
        base = torch.arange(10,dtype=torch.float32).reshape(1,5,2)
        current = torch.tensor([[5.,6.,7.]])
        hook,trace = additive_hook(author,'white pillow','bed',.1)
        result = hook(base,current,'original caption',torch.ones(1,3),torch.ones(1,2),1,model)
        self.assertTrue(torch.equal(result[:,:-1,:],base[:,:-1,:]))
        self.assertTrue(torch.allclose(result[:,-1,:],base[:,-1,:]+torch.tensor([[.6,.7]])))
        self.assertTrue(torch.equal(current,torch.tensor([[5.,6.,7.]])))
        zero,_ = additive_hook(author,'white pillow','bed',0)
        self.assertIs(zero(base,current,'text',torch.ones(1,3),None,1,model),base)
        self.assertTrue(trace[0]['original_image_condition_preserved'])

    def test_identity_leak_is_error(self):
        with self.assertRaises(ValueError):
            additive_hook(None,'white bed','bed',.1)

    def test_offline_rounds_not_training_epochs(self):
        self.assertEqual(coefficient_for_round(.05,.15,0,3),.05)
        self.assertAlmostEqual(coefficient_for_round(.05,.15,2,3),.15)
        row = dict(path='source.jpg',category='car',caption='a wheel')
        other = dict(path='other.jpg',category='car',caption='a window')
        self.assertEqual(len(merge_pool([row],[[row],[other]])),2)
        self.assertEqual(len(bank_candidates([[row],[other]],'last')),1)
        self.assertEqual(len(bank_candidates([[row],[other]],'all')),2)


if __name__=='__main__':
    unittest.main()
