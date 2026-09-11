import csv
from dataclasses import asdict
import json
import pytest
import torch
from PIL import Image
from ncap.config import TrainConfig
from ncap.losses import excess_state_loss
from ncap.objective_study import run_objective_study, summarize_objective
from ncap.simulate import load_checkpoint


def test_excess_penalty_value_and_gradients():
    state=torch.tensor([3.,-3.,2.,0.]).reshape(1,4,1,1).requires_grad_()
    loss=excess_state_loss(state,2)
    assert loss.item()==.5
    loss.backward()
    assert torch.equal(state.grad.flatten(),torch.tensor([.5,-.5,0.,0.]))
    hidden=torch.tensor([0.,0.,0.,0.,4.]).reshape(1,5,1,1).requires_grad_()
    excess_state_loss(hidden,2).backward()
    assert hidden.grad[0,4,0,0].item()==pytest.approx(.8)


@pytest.mark.parametrize('values',[{'excess_weight':-1},{'excess_weight':float('nan')},
                                  {'excess_threshold':.5},{'excess_threshold':float('inf')}])
def test_invalid_objective_config(values):
    with pytest.raises(ValueError):
        TrainConfig(**values)


def test_objective_study_logs_and_unclamped_checkpoint(tmp_path):
    target=tmp_path/'target.png'; Image.new('RGBA',(1,1),(0,150,0,255)).save(target)
    plan=dict(training=asdict(TrainConfig(channels=4,hidden_size=8,size=5,padding=2,
               batch_size=2,iterations=5,min_steps=2,max_steps=3,eval_steps=3,pool_size=2,
               damage_probability=.5)),training_seeds=[0],evaluation_seeds=[10],horizons=[3,8],excess_weight=.01)
    output=tmp_path/'study'; summary=run_objective_study(target,output,plan)
    assert len(summary)==8
    baseline=json.loads((output/'baseline-0/config.json').read_text())
    regularized=json.loads((output/'regularized-0/config.json').read_text())
    assert {k for k in baseline if baseline[k]!=regularized[k]}=={'excess_weight'}
    with (output/'regularized-0/loss.csv').open() as stream:
        for row in csv.DictReader(stream):
            assert float(row['loss'])==pytest.approx(float(row['image_loss'])+.01*float(row['excess_loss']))
    with (output/'baseline-0/loss.csv').open() as stream:
        assert all(row['loss']==row['image_loss'] for row in csv.DictReader(stream))
    model,_=load_checkpoint(output/'regularized-0/checkpoint.pt')
    assert model.state_limit is None
    with pytest.raises(FileExistsError):
        run_objective_study(target,output,plan)


def test_training_seed_pair_summary():
    rows=[]
    for seed,count,value in [(0,1,2),(1,3,6)]:
        for _ in range(count):
            for arm in ['baseline','regularized']:
                v=value if arm=='baseline' else value/2
                rows.append(dict(step=10,training_seed=seed,arm=arm,loss=v,alpha_iou=v,
                                 foreground_cells=v,state_abs_max=v))
    row=summarize_objective(rows)[0]
    assert row['baseline_mean']==4
    assert row['regularized_mean']==2
