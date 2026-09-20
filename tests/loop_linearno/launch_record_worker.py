"""LL8 opt-in records on the unchanged LL6/7 native CPU synthetic workers."""
import runpy
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
from tran_evaluate.linearno_loop.recording import install
install()
task,*arguments=sys.argv[1:]
worker=ROOT/'tests/loop_linearno'/({'airfrans':'air_worker.py','car':'car_worker.py'}.get(task,'native_worker.py'))
sys.argv=[str(worker),*(([task]+arguments) if task not in ('car','airfrans') else arguments)]
runpy.run_path(str(worker),run_name='__main__')
