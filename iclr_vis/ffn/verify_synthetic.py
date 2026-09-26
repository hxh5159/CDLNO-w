#!/usr/bin/env python3
"""Optional local acceptance tool. Creates only synthetic fixtures under /tmp.

PyMuPDF is optional for users of the plotting scripts, required for this PDF audit.
The output report records exact artifact paths; no files in a training run change.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import fitz

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import ffn_states as ffn
from test_ffn_states import air_fixture,task_fixture,nonuniform


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    from cdlno.linearno_loop.v5.checkpoint import save_pair
    from linearno_loop.v5.schema import write_metadata
    torch.set_num_threads(1)
    root=Path(tempfile.mkdtemp(prefix='v5-ffn-visual-'))
    report=dict(status='RUNNING',scope='synthetic inputs/checkpoints only',artifact_root=str(root),tasks=[])
    try:
        for task in ffn.TASKS:
            folder=root/task;folder.mkdir()
            if task=='airfoil':run,data,meta,model,_=air_fixture.make_fixture(folder)
            else:run,data,meta,model=task_fixture.make_fixture(folder,task)
            # Separate synthetic archive with nonzero routers; preserve initial fixture.
            nonuniform(model)
            selected=folder/'nonuniform saved run';selected.mkdir()
            write_metadata(selected/'architecture.json',meta);save_pair(selected,model,meta)
            before={str(f):ffn.vis.sha256(f) for f in selected.rglob('*') if f.is_file()}
            out=folder/'figures with spaces'
            command=['bash',str(HERE/(task+'.sh')),'--run-dir',str(selected),'--data-path',str(data),
                '--output-dir',str(out),'--device','cpu','--font-family','STIXGeneral','--dpi','100',
                '--paper','icml' if task=='darcy' else 'iclr']
            if task=='ns':command+=['--forecast-step','6']
            env={**os.environ,'CDLNO_PYTHON':sys.executable,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            print('COMMAND '+repr(command),flush=True)
            subprocess.run(command,cwd='/tmp',env=env,check=True)
            m=json.loads((out/'metadata.json').read_text())
            assert m['status']=='completed' and m['inference']['diagnostic']['hooked_prediction_bitwise_equal']
            assert before=={str(f):ffn.vis.sha256(f) for f in selected.rglob('*') if f.is_file()}
            with np.load(out/'expert_weights.npz',allow_pickle=False) as a:
                assert a['probabilities'].shape[0]==8
                np.testing.assert_allclose(a['probabilities'].sum(-1),1,atol=5e-6)
                for key in a.files:
                    if np.issubdtype(a[key].dtype,np.number):assert np.isfinite(a[key]).all(),key
                if task=='darcy':
                    np.testing.assert_array_equal(a['probabilities'],np.ones_like(a['probabilities']))
                    assert 'normalized_entropy' not in a
                else:assert np.ptp(a['probabilities'])>.01
            pdfs=[]
            for path in sorted(out.glob('*.pdf')):
                doc=fitz.open(path);page=doc[0]
                assert abs(page.rect.width/72-m['rendering']['width_inches'])<1e-5
                spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
                assert spans,path
                assert all(abs(s['size']-8)<1e-5 or abs(s['size']-9)<1e-5 for s in spans),path
                for span in spans:
                    x0,y0,x1,y1=span['bbox']
                    assert x0>=-.05 and y0>=-.05 and x1<=page.rect.width+.05 and y1<=page.rect.height+.05,(path,span)
                fonts=page.get_fonts(full=True)
                assert fonts and all('STIXGeneral' in font[3] and doc.extract_font(font[0])[3] for font in fonts),(path,fonts)
                pdfs.append(dict(file=path.name,width_inches=page.rect.width/72,
                    font_sizes=sorted(set(round(s['size'],4) for s in spans)),fonts=[f[3] for f in fonts],text_within_page=True))
                doc.close()
            expected=38 if task=='airfoil' else (15 if task=='darcy' else 20)
            assert len(pdfs)==expected,(task,len(pdfs),expected)
            # Explicit existing output refusal is an end-to-end shell failure check.
            rerun=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True)
            assert rerun.returncode!=0 and 'already exists' in rerun.stderr
            report['tasks'].append(dict(task=task,output_dir=str(out),pdf_count=len(pdfs),pdfs=pdfs,
                                       original_run_hashes_unchanged=True,existing_output_rejected=True))
            args.report.write_text(json.dumps(report,indent=2)+'\n')
        report.update(status='PASS',pdf_count=sum(t['pdf_count'] for t in report['tasks']))
    except Exception as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(f"PASS: five tasks, {report['pdf_count']} PDFs, unchanged checkpoints, same forward predictions.",flush=True)


if __name__=='__main__':main()
