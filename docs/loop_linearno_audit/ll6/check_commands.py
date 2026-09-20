"""Preview unchanged launchers and parse their exact argv without entry imports."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tests'),str(ROOT)]
from loop_linearno.test_standard_entry import parse
from linearno_loop.config import resolve_config,run_directory_id
from linearno_loop.contracts import PRESETS,RESIDUAL_MODES


def main():
    env=dict(os.environ,CDLNO_REPO_ROOT=str(ROOT),CDLNO_RUNS_ROOT=str(ROOT/'output'),
             PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    rows=[]
    def check(task,action,arguments,config):
        cmd=['bash',f'tran_evaluate/linearno/{task}_{action}.sh','--dry-run',*arguments]
        text=subprocess.check_output(cmd,cwd=ROOT,env=env,text=True)
        line=next(s for s in text.splitlines() if s.startswith('Command:'))
        argv=shlex.split(line[len('Command:'):])[3:]
        args=parse(task,argv)
        assert args._linearno_loop_config==config
        assert args.linearno_family=='linearno_loop'
        rows.append(dict(task=task,action=action,command=cmd,preview=text,
                         config_hash=config['config_hash'],pass_=True))
    for task in ('airfoil','darcy','elasticity','pipe','ns','plasticity'):
        for profile in ('paper_table8_on_release_model','official_release'):
            for preset in PRESETS:
                for mode in RESIDUAL_MODES:
                    config=resolve_config(task,profile,options=dict(topology_preset=preset,
                        residual_mode=mode,rank_multiplier=2),profile_overrides={'runtime.seed':0})
                    directory=ROOT/'output'/task/'linearno_loop'/(run_directory_id(config)+'__preview_only')
                    check(task,'train',['--gpu','0','--seed','0','--experiment-dir',str(directory),
                        '--linearno-profile',profile,'--linearno-loop','1','--linearno-loop-topology',preset,
                        '--linearno-loop-residual-mode',mode,'--linearno-loop-rank-multiplier','2'],config)
                    assert not directory.exists()
    matrix=json.loads((OUT/'native-matrix.json').read_text())
    assert matrix['completed']==matrix['expected']==36
    for row in matrix['rows']:
        directory=next((Path(row['artifact'])/'split').iterdir())
        config=json.loads((directory/'architecture.json').read_text())['resolved_config']
        check(row['task'],'eval',['--gpu','1','--experiment-dir',str(directory)],config)
    (OUT/'command-previews.json').write_text(json.dumps(dict(train=72,eval=36,rows=rows),indent=2)+'\n')
    print('72 train and 36 saved-run eval launcher previews + actual parsers passed; no entry/data execution')


if __name__=='__main__':main()
