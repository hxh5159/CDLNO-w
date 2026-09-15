"""Preview eight tasks x three modes; parse commands without importing entries."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
import test_front_task_modes as helpers
from cdlno import CDLNOArchitectureConfig
from cdlno.checkpoint import save_sidecar

records=[]
for task in helpers.TASKS:
    for mode in helpers.MODES:
        command=['bash','tran_evaluate/train_eval.sh',task,'--front-latent-mode',mode,'--dry-run']
        text=subprocess.check_output(command,cwd=ROOT,text=True,env=dict(os.environ,CDLNO_RUN_TAG='a4_preview'))
        commands=[shlex.split(s.removeprefix('Command:')) for s in text.splitlines() if s.startswith('Command:')]
        assert len(commands)==2
        paths=[]
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            save_sidecar(Path(tmp)/'architecture.json',CDLNOArchitectureConfig(front_latent_mode=mode))
            for evaluation,tokens in enumerate(commands):
                argv=tokens[3:]
                flag='--cdlno-run-dir' if task in helpers.STANDARD else '--run_dir'
                paths.append(argv[argv.index(flag)+1])
                argv+=helpers.run_option(task,tmp)
                if task in helpers.STANDARD:
                    p=helpers.static.parser_for(task) if task in helpers.static.TASKS else helpers.temporal.parser(task)
                    args=helpers.static.parse_args(p,task,argv)
                elif task=='car':
                    args=helpers.car.parse_args(helpers.car.entry_parser(bool(evaluation)),evaluation=bool(evaluation),argv=argv)
                else:
                    args=helpers.air.entry.parse_args(helpers.air.parser(bool(evaluation)),evaluation=bool(evaluation),argv=argv)
                assert args.front_latent_mode==mode
                assert args.slice_num==(32 if task=='pipe' else 64)
        assert paths[0]==paths[1]
        if mode!='full': assert paths[0].endswith('_'+mode)
        records.append(dict(task=task,mode=mode,preview=command,commands=commands,run=paths[0],
                            actual_parser_passed=True,temporary_sidecar_only=True))
assert not any(n.startswith('exp_') for n in sys.modules)
(OUT/'commands.json').write_text(json.dumps(dict(status='passed',pairs=len(records),commands=2*len(records),records=records),indent=2)+'\n')
print('PASS: 24 train/eval previews; 48 actual parser checks; no data/entry execution')
