import copy
from pathlib import Path
import tempfile
import unittest

import torch

from cdlno.linearno.profiles import DEFAULT_PROFILE
from cdlno.linearno.standard_entry import finish, model_kwargs, start
from cdlno.linearno.schema import unpack_state
from cdlno.linearno_loop.standard_entry import LoopStandardRun
from cdlno.linearno_loop.v2 import checkpoint
from linearno_loop.contracts import PRESETS, RESIDUAL_MODES, digest
from linearno_loop.v2.config import resolve_config, run_directory_id
from linearno_loop.v2.contracts import CORE_FFN_MODES
from linearno.static_worker import parser_for
from linearno.temporal_worker import parser_for as temporal_parser
from cdlno_entry import parse_args
from model_dict import get_model
from linearno_entry import StandardRun

TASKS=("airfoil","darcy","elasticity","pipe","ns","plasticity")


def parse(task,tokens):
    parser=(temporal_parser if task in ("ns","plasticity") else parser_for)(task)
    return parse_args(parser,task,list(map(str,tokens)))


def flags(task,preset,residual,mode):
    model="LinearNO_Irregular_Mesh" if task=="elasticity" else "LinearNO_Structured_Mesh_2D"
    return ["--model",model,"--linearno-loop","1","--linearno-loop-topology",preset,
            "--linearno-loop-residual-mode",residual,"--linearno-loop-core-ffn-mode",mode]


class LF5StandardEntryTests(unittest.TestCase):
    def test_six_task_real_parsers_two_presets_three_residuals_two_modes(self):
        for task in TASKS:
            for preset in PRESETS:
                for residual in RESIDUAL_MODES:
                    for mode in CORE_FFN_MODES:
                        args=parse(task,flags(task,preset,residual,mode))
                        config=args._linearno_loop_config
                        self.assertEqual(config["config_version"],2)
                        self.assertEqual(config["loop_spec"]["core_ffn_mode"],mode)
                        self.assertEqual(args.linearno_rank,config["loop_spec"]["base_rank"])
                        self.assertEqual(model_kwargs(args),config["model_spec"]["constructor_kwargs"])
                        self.assertIn(run_directory_id(config),args.linearno_run_dir.name)

    def test_explicit_only_selection_and_v1_default_unchanged(self):
        old=parse("darcy",["--model","LinearNO_Structured_Mesh_2D","--linearno-loop","1",
            "--linearno-loop-topology","p2_c2_r2_s2","--linearno-loop-residual-mode","sr_1_over_r"])
        self.assertEqual(old._linearno_loop_config["config_version"],1)
        self.assertEqual(old.linearno_rank,2*old._linearno_loop_config["loop_spec"]["base_rank"])
        new=parse("darcy",flags("darcy","p2_c2_r2_s2","sr_1_over_r","round_specific"))
        self.assertEqual(new.linearno_rank,new._linearno_loop_config["loop_spec"]["base_rank"])

    def test_production_standard_checkpoint_resume_eval_metadata_first(self):
        task="elasticity";base_flags=flags(task,"p1_c3_r2_s1","lb_attnres_1_over_r","round_specific_latent")
        tuning=["--epochs",3,"--n-hidden",8,"--n-heads",2,"--linearno-rank",4,
                "--batch-size",2,"--seed",17,"--dropout",0.0]
        expected=resolve_config(task,DEFAULT_PROFILE,options={"topology_preset":"p1_c3_r2_s1",
            "residual_mode":"lb_attnres_1_over_r","core_ffn_mode":"round_specific_latent","linearno_rank":4},
            profile_overrides={"training.epochs":3,"model.hidden":8,"model.heads":2,
                "training.batch_size":2,"runtime.seed":17,"model.dropout":0.0})
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/run_directory_id(expected)
            args=parse(task,[*base_flags,*tuning,"--experiment-dir",directory])
            self.assertEqual(args._linearno_loop_config,expected)
            args._linearno_data=dict(split=expected["profile_spec"]["values"]["data"]["split"],
                sampling=expected["profile_spec"]["values"]["data"]["sampling"],
                checksums={"SYNTHETIC":digest("lf5")},scope="SYNTHETIC")
            start(args,task)
            try:
                model=get_model(args).Model(**model_kwargs(args));run=StandardRun(args,model)
                self.assertIsInstance(run,LoopStandardRun)
                x=torch.randn(4,9,2)
                train=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x),batch_size=2,shuffle=True)
                test=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[:2]),batch_size=2)
                optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
                scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,epochs=3,steps_per_epoch=2)
                run.prepare(optimizer,scheduler,train,test)
                for (batch,) in train:
                    optimizer.zero_grad();model(batch,None).square().mean().backward();optimizer.step();scheduler.step()
                run.complete_epoch(1);manifest=run.save(model)
            finally:
                finish(args)
            saved,weights=checkpoint.read_pair(manifest,expected=expected)
            self.assertEqual(saved["architecture_extension"],"loop_linearno_ffn_v2")
            resume=parse(task,["--resume","--experiment-dir",directory])
            self.assertEqual(resume._linearno_loop_config,expected)
            resume._linearno_data=copy.deepcopy(args._linearno_data);start(resume,task)
            try:
                resumed=get_model(resume).Model(**model_kwargs(resume))
                train2=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x),batch_size=2,shuffle=True)
                test2=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x[:2]),batch_size=2)
                optimizer2=torch.optim.AdamW(resumed.parameters(),lr=resume.lr,weight_decay=resume.weight_decay)
                scheduler2=torch.optim.lr_scheduler.OneCycleLR(optimizer2,max_lr=resume.lr,epochs=3,steps_per_epoch=2)
                run2=LoopStandardRun(resume,resumed);run2.prepare(optimizer2,scheduler2,train2,test2)
                self.assertEqual(run2.start_epoch,1)
                self.assertTrue(all(torch.equal(value,resumed.state_dict()[key]) for key,value in weights.items()))
            finally:
                finish(resume)
            eval_args=parse(task,["--eval",1,"--experiment-dir",directory,"--checkpoint","latest"])
            self.assertEqual(eval_args._linearno_loop_config,expected)
            with self.assertRaises(SystemExit):
                parse(task,["--resume","--experiment-dir",directory,
                    "--linearno-loop-core-ffn-mode","round_specific"])


if __name__=="__main__":unittest.main()
