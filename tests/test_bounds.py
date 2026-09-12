from dataclasses import asdict
import json
import pytest
import torch
from PIL import Image
from ncap.config import TrainConfig
from ncap.model import NeuralCellularAutomata
from ncap.state import create_seed
from ncap.simulate import load_checkpoint
from ncap.bound_study import run_bound_study, summarize_bounds


def test_bound_applies_to_all_channels_and_preserves_dead_cells():
    model=NeuralCellularAutomata(channels=8,hidden_size=8,fire_rate=1,state_limit=2)
    state=torch.ones(1,8,5,5)
    with torch.no_grad():
        model.update[-1].bias.fill_(100)
    result=model.rollout(state,10)
    assert result.abs().max()==2
    assert torch.isfinite(result).all()
    assert model(torch.zeros_like(state)).count_nonzero()==0
    assert state.eq(1).all()


def test_bound_keeps_gradients_inside_range():
    model=NeuralCellularAutomata(channels=4,hidden_size=8,fire_rate=1,state_limit=2)
    state=torch.ones(1,4,3,3,requires_grad=True)
    with torch.no_grad():
        model.update[-1].bias.fill_(.1)
    result=model(state)
    result.sum().backward()
    assert torch.all(state.grad==1)
    assert model.update[-1].weight.grad.abs().sum()>0
    assert torch.allclose(result,torch.full_like(state,1.1))


@pytest.mark.parametrize('limit',[0,.5,float('nan'),float('inf'),True])
def test_invalid_limits(limit):
    with pytest.raises(ValueError):
        NeuralCellularAutomata(state_limit=limit)
    with pytest.raises(ValueError):
        TrainConfig(state_limit=limit)


def test_old_checkpoint_stays_unbounded(tmp_path):
    config=asdict(TrainConfig(channels=4,hidden_size=8))
    config.pop('state_limit')
    model=NeuralCellularAutomata(4,8)
    path=tmp_path/'old.pt'
    torch.save(dict(format_version=1,config=config,model=model.state_dict()),path)
    loaded,_=load_checkpoint(path)
    assert loaded.state_limit is None
    assert torch.equal(loaded(create_seed(channels=4)),create_seed(channels=4))


def test_bounded_study_replay_and_matching(tmp_path):
    target=tmp_path/'target.png'; Image.new('RGBA',(1,1),(0,150,0,255)).save(target)
    plan=dict(training=asdict(TrainConfig(channels=4,hidden_size=8,size=5,padding=2,
               batch_size=2,iterations=3,min_steps=2,max_steps=3,eval_steps=3,pool_size=2,
               damage_probability=.5)),training_seeds=[0],evaluation_seeds=[10,11],
               horizons=[3,8],state_limit=2)
    output=tmp_path/'study'
    summary=run_bound_study(target,output,plan)
    assert len(summary)==8
    unbounded=json.loads((output/'unbounded-0/config.json').read_text())
    bounded=json.loads((output/'bounded-0/config.json').read_text())
    assert {k for k in unbounded if unbounded[k]!=bounded[k]}=={'state_limit'}
    loaded,_=load_checkpoint(output/'bounded-0/checkpoint.pt')
    assert loaded.state_limit==2
    model_rows=json.loads((output/'eval-bounded-0/metrics.json').read_text())
    state=loaded.rollout(create_seed(channels=4,height=5,width=5),8,
                         generator=torch.Generator().manual_seed(10))
    assert model_rows[1]['state_abs_max']==state.abs().max().item()
    assert all(r['state_abs_max']<=2 for r in model_rows)
    with pytest.raises(FileExistsError):
        run_bound_study(target,output,plan)


def test_paired_summary_averages_models_first():
    rows=[]
    for seed,count,value in [(0,1,2),(1,3,6)]:
        for _ in range(count):
            for arm in ['unbounded','bounded']:
                v=value if arm=='unbounded' else value/2
                rows.append(dict(step=10,training_seed=seed,arm=arm,loss=v,alpha_iou=v,
                                 foreground_cells=v,state_abs_max=v))
    row=summarize_bounds(rows)[0]
    assert row['unbounded_mean']==4
    assert row['bounded_mean']==2
