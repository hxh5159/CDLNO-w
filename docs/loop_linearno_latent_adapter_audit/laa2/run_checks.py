"""LAA2 test runner with raw logs and numerical-error evidence. No real data."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SUITES={
    'new': [f'loop_linearno_latent_adapter.test_laa2_{name}' for name in ('oracles','adapter','latent','rng_amp')],
    'schema': [f'loop_linearno_latent_adapter.test_laa1_{name}' for name in ('config','costs','schema','isolation')],
    'regression': ['linearno.test_attention_parity','linearno.test_attention_structure',
                   'loop_linearno.test_point_attnres','loop_linearno.test_sr_core',
                   'loop_linearno.test_rb_core','loop_linearno.test_lb_core','loop_linearno.test_ll9r',
                   'loop_linearno_ffn.test_lf2_core','loop_linearno_ffn.test_lf3_latent'],
}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('suite',choices=SUITES);parser.add_argument('--label',default='final')
    args=parser.parse_args()
    sys.path[:0]=[str(ROOT/'tests'),str(ROOT/'tests/loop_linearno_ffn'),str(ROOT)]
    import torch
    torch.set_num_threads(1)
    start=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromNames(SUITES[args.suite])
    stem=args.suite+'-'+args.label
    with (OUT/(stem+'.log')).open('x') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    elapsed=time.perf_counter()-start
    record=dict(stage='LAA2',suite=args.suite,command=[sys.executable,'-B',*sys.argv],cwd=str(Path.cwd()),
                modules=SUITES[args.suite],status='PASS' if result.wasSuccessful() else 'FAIL',
                tests=result.testsRun,failed=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                skip_details=[dict(test=str(t),reason=r) for t,r in result.skipped],wall_seconds=elapsed,
                log=stem+'.log',environment=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
                    torch=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),
                    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    packages={p:importlib.metadata.version(p) for p in ('numpy','torch-geometric','timm')},
                    threads=torch.get_num_threads(),env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}))
    if args.suite=='new':
        from loop_linearno_latent_adapter.primitive_support import ERRORS
        from loop_linearno_latent_adapter.test_laa2_rng_amp import GPU_ROWS
        record['cpu_error_rows']=ERRORS;record['cuda_cases']=GPU_ROWS
    with (OUT/(stem+'.json')).open('x') as stream:json.dump(record,stream,indent=2)
    print(json.dumps({k:v for k,v in record.items() if k not in ('environment','cpu_error_rows','modules','skip_details')}))
    raise SystemExit(not result.wasSuccessful())


if __name__=='__main__':main()
