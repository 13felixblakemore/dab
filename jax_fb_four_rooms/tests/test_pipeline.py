from pathlib import Path
import json
from fb_jax.experiment import load_configuration,train
from fb_jax.agent import FBAgent
from fb_jax.navigation import evaluate_navigation


CONFIG=Path(__file__).resolve().parents[1]/"configs"/"smoke.yaml"


def test_config_and_end_to_end(tmp_path):
    _,features,model,trainer=load_configuration(CONFIG)
    assert model.embed_dim==8 and features.output_dim==25 and trainer.epochs==2
    out,summary=train(CONFIG,tmp_path/"run",verbose=False)
    assert summary["updates"]==4
    for filename in ("training_metrics.csv","resolved_config.yaml","summary.json",
                     "fb_final.npz","training_curve.png","fb_epoch_0001.npz"):
        assert (out/filename).stat().st_size>0
    saved=FBAgent.load(out/"fb_final.npz")
    evaluation,example,env=evaluate_navigation(saved,episodes=2,max_steps=5,seed=30)
    assert 0<=evaluation["success_rate"]<=1
    assert example["states"].shape[1]==2
    assert example["goal"].shape==(2,)
    assert len(json.loads((out/"summary.json").read_text()))>=4
