"""M4 safe example using the real PDE factory; synthetic LIFTED inputs only.

No exp/main imports, task training, data files, task adapters, or fake datasets.
Without --output-root, creates no checkpoint directory. With it, uses a fresh
family/profile/coverage path and never replaces an existing experiment.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'PDE-Solving-StandardBenchmark'))

import torch
from model_dict import get_model
from utils.testloss import TestLoss
from cdlno.msar_lno.options import parser_for_family,explicit_arguments,resolve_training
from cdlno.msar_lno.entry import core_model_kwargs
from cdlno.msar_lno.objective import training_forward,training_objective
from cdlno.msar_lno.config import input_layout
from cdlno.msar_lno.metadata import MSARMetadata,new_run_path
from cdlno.msar_lno.checkpoint import save_core_checkpoint,load_core_checkpoint


def main():
    base=argparse.ArgumentParser(description=__doc__)
    base.add_argument('--model',choices=('msar_lno',),default='msar_lno')
    base.add_argument('--output-root',type=Path)
    base.add_argument('--save-name')
    parser=parser_for_family(base,'msar_lno')
    args=parser.parse_args();explicit=explicit_arguments(parser,sys.argv[1:])
    resolved=resolve_training('darcy',explicit)
    torch.set_num_threads(1);torch.manual_seed(916404)
    model=get_model(args).Model(**core_model_kwargs('msar_lno',task='darcy',explicit=explicit,output_dim=1))
    x=torch.randn(2,35,resolved.architecture.d)
    target=torch.randn(2,35,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    result=training_forward(model.train(),x)
    # The exact safe repository loss, applied to synthetic outputs. Darcy's
    # decoder/boundary/gradient loss additions and actual loop are not run here.
    pde=TestLoss(size_average=False)(result.prediction,target)
    loss=training_objective(pde,result)
    loss.total.backward();optimizer.step()
    record=dict(scope='synthetic lifted-core M4 construction; not a Darcy task training loop',
                resolved=resolved.to_dict(),layout=input_layout(model.config,x.shape[1]),
                parameters=sum(p.numel() for p in model.parameters()),
                objective={key:float(value) for key,value in loss.log_values().items()},
                prediction_shape=list(result.prediction.shape))
    with torch.no_grad():
        expected=model.eval()(x)
        if args.output_root is not None:
            directory=new_run_path(args.output_root,resolved,save_name=args.save_name)
            metadata=MSARMetadata('darcy',model.config,'state_dict',resolved.profile,training=resolved.training)
            save_core_checkpoint(directory,model,metadata)
            loaded=load_core_checkpoint(directory,task='darcy',explicit={'coverage_mode':'off'}).model
            torch.testing.assert_close(loaded(x),expected,atol=0,rtol=0)
            record['checkpoint']=str(directory)
            record['same_weight_eval']='exact'
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
