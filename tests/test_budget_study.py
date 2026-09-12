from dataclasses import asdict,replace
import json
import pytest
import torch
from PIL import Image
from ncap.config import TrainConfig
from ncap.trainer import train
from ncap.budget_study import run_budget_study,summarize_budgets


def config():
    return TrainConfig(channels=4,hidden_size=8,size=5,padding=2,batch_size=2,
                       iterations=6,min_steps=2,max_steps=3,eval_steps=3,pool_size=2,
                       damage_probability=.5)


def test_checkpoint_prefix_matches_standalone_and_does_not_change_training(tmp_path):
    target=tmp_path/'target.png'; Image.new('RGBA',(1,1),(0,150,0,255)).save(target)
    cfg=config()
    train(target,tmp_path/'full',cfg,checkpoint_iterations=[3])
    train(target,tmp_path/'short',replace(cfg,iterations=3))
    train(target,tmp_path/'plain',cfg)
    milestone=torch.load(tmp_path/'full/checkpoint-3.pt',weights_only=True)
    short=torch.load(tmp_path/'short/checkpoint.pt',weights_only=True)
    final=torch.load(tmp_path/'full/checkpoint.pt',weights_only=True)
    plain=torch.load(tmp_path/'plain/checkpoint.pt',weights_only=True)
    for a,b in [(milestone,short),(final,plain)]:
        for name in a['model']:
            assert torch.equal(a['model'][name],b['model'][name])
        for key in ['pool','generator_state','pool_generator_state','damage_generator_state']:
            assert torch.equal(a[key],b[key])
        assert a['python_rng_state']==b['python_rng_state']
    assert milestone['iteration']==milestone['config']['iterations']==3
    assert milestone['planned_iterations']==6
    assert milestone['training_seconds']<=final['training_seconds']
    assert (tmp_path/'full/loss.csv').read_bytes()==(tmp_path/'plain/loss.csv').read_bytes()


def test_fixed_budget_study(tmp_path):
    target=tmp_path/'target.png'; Image.new('RGBA',(1,1),(0,150,0,255)).save(target)
    plan=dict(training=asdict(config()),training_seeds=[0,1],evaluation_seeds=[10],horizons=[3,8],budgets=[3,6])
    out=tmp_path/'study'; result=run_budget_study(target,out,plan)
    assert len(result)==16
    assert len(json.loads((out/'rows.json').read_text()))==8
    costs=json.loads((out/'costs.json').read_text())
    for seed in [0,1]:
        first,last=[r for r in costs if r['training_seed']==seed]
        assert first['sample_updates']<last['sample_updates']
        assert first['training_seconds']<=last['training_seconds']
    assert all(r['paired_delta_mean']==0 for r in result if r['budget']==3)
    with pytest.raises(FileExistsError):
        run_budget_study(target,out,plan)


def test_budget_summary_pairs_training_seeds():
    rows=[]
    for seed,count,value in [(0,1,2),(1,3,6)]:
        for _ in range(count):
            for budget in [3,6]:
                v=value if budget==3 else value/2
                rows.append(dict(step=10,training_seed=seed,budget=budget,loss=v,alpha_iou=v,
                                 foreground_cells=v,state_abs_max=v))
    row=next(r for r in summarize_budgets(rows) if r['budget']==6 and r['metric']=='loss')
    assert row['mean']==2
    assert row['paired_delta_mean']==-2


@pytest.mark.parametrize('points',[[0],[6],[3,3],[-1],[2.5]])
def test_invalid_checkpoint_schedule_is_rejected_before_writing(tmp_path,points):
    with pytest.raises(ValueError):
        train(tmp_path/'missing.png',tmp_path/'run',config(),checkpoint_iterations=points)
    assert not (tmp_path/'run').exists()
