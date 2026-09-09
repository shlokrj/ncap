from dataclasses import asdict
import json
import pytest
from PIL import Image
from ncap.config import TrainConfig
from ncap.horizon_study import run_horizon_study, validate_plan, summarize_horizons


def small_plan():
    return dict(training=asdict(TrainConfig(channels=4, hidden_size=8, size=5,padding=2,
                batch_size=2,iterations=3,min_steps=1,max_steps=2,eval_steps=3,pool_size=2,
                damage_probability=.5)),training_seeds=[0,1],evaluation_seeds=[10],horizons=[2,6],
                arms=dict(short=dict(min_steps=1,max_steps=2),long=dict(min_steps=3,max_steps=4)))


def test_only_rollout_range_can_change():
    plan=small_plan()
    plan['arms']['long']['learning_rate']=.1
    with pytest.raises(ValueError, match='only'):
        validate_plan(plan)
    plan=small_plan(); plan['arms']['long']['min_steps']=2
    with pytest.raises(ValueError, match='beyond'):
        validate_plan(plan)
    plan=small_plan(); plan['horizons']=[2,4]
    with pytest.raises(ValueError, match='beyond'):
        validate_plan(plan)


def test_horizon_study_and_costs(tmp_path):
    target=tmp_path/'target.png'
    Image.new('RGBA',(1,1),(0,150,0,255)).save(target)
    output=tmp_path/'study'
    summary=run_horizon_study(target,output,small_plan())
    assert len(summary)==6
    assert len(json.loads((output/'rows.json').read_text()))==8
    short=json.loads((output/'short-0/config.json').read_text())
    long=json.loads((output/'long-0/config.json').read_text())
    assert {k for k in short if short[k]!=long[k]}=={'min_steps','max_steps'}
    costs=json.loads((output/'costs.json').read_text())
    assert all(3<=r['rollout_updates']<=6 for r in costs if r['arm']=='short')
    assert all(9<=r['rollout_updates']<=12 for r in costs if r['arm']=='long')
    assert all(r['sample_updates']==r['rollout_updates']*2 for r in costs)
    assert json.loads((output/'status.json').read_text())['status']=='complete'
    with pytest.raises(FileExistsError):
        run_horizon_study(target,output,small_plan())


def test_aggregation_uses_training_seed_pairs():
    rows=[]
    for seed,count,value in [(0,1,2),(1,3,6)]:
        for _ in range(count):
            for arm in ['short','long']:
                v=value if arm=='short' else value/2
                rows.append(dict(step=10,training_seed=seed,arm=arm,loss=v,alpha_iou=v,foreground_cells=v))
    summary=summarize_horizons(rows)
    assert summary[0]['short_mean']==4
    assert summary[0]['long_mean']==2
    assert summary[0]['paired_delta_mean']==-2


def test_failed_training_keeps_plan(tmp_path,monkeypatch):
    from ncap import horizon_study
    target=tmp_path/'target.png'; Image.new('RGBA',(1,1)).save(target)
    def fail(*args,**kwargs):
        raise RuntimeError('test failure')
    monkeypatch.setattr(horizon_study,'train',fail)
    with pytest.raises(RuntimeError):
        run_horizon_study(target,tmp_path/'study',small_plan())
    assert json.loads((tmp_path/'study/status.json').read_text())['status']=='failed'
    assert (tmp_path/'study/plan.json').exists()
