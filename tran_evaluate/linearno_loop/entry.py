"""Explicit recording bridge into the unchanged native benchmark entry."""
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tran_evaluate.linearno_loop.launch import project,ENTRIES,TASKS


def main():
    task,action,entry,*tokens=sys.argv[1:]
    expected=ENTRIES.get(task) or ('main_evaluation.py' if action=='eval' else 'main.py')
    if task not in TASKS or action not in ('train','resume','eval') or entry!=expected or Path.cwd()!=project(task):
        raise ValueError('invalid native loop entry target')
    from tran_evaluate.linearno_loop.recording import install
    install()
    sys.path.insert(0,str(Path.cwd()))
    sys.argv=[entry,*tokens]
    runpy.run_path(entry,run_name='__main__')


if __name__=='__main__':main()
