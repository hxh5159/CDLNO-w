"""M9 final integrity/result index; no model, data or task entry imports."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,data):(E/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


class OriginalTimingDefaults(ast.NodeTransformer):
    """Project only the explicitly added optional callback; no broad stripping."""
    def visit_FunctionDef(self,node):
        node=self.generic_visit(node)
        if node.name=='benchmark':
            assert node.args.kwonlyargs[-1].arg=='loss_closure'
            assert isinstance(node.args.kw_defaults[-1],ast.Constant) and node.args.kw_defaults[-1].value is None
            node.args.kwonlyargs.pop();node.args.kw_defaults.pop()
        return node
    def visit_If(self,node):
        condition=ast.unparse(node.test)
        if condition=='loss_closure is not None and compile_model':
            assert len(node.body)==1 and isinstance(node.body[0],ast.Raise)
            assert 'custom training loss timing supports eager models only' in ast.unparse(node.body[0])
            return None
        if condition=='loss_closure is None':
            assert ast.unparse(node.orelse[0])=='loss = loss_closure(run_model, args, target)'
            assert len(node.orelse)==1
            return node.body
        return self.generic_visit(node)
    def visit_IfExp(self,node):
        if ast.unparse(node.test)=='loss_closure is None':
            assert isinstance(node.body,ast.Constant)
            assert node.body.value=='synthetic FP32 mean squared error, one model call/update; not task loss/temporal rollout'
            return node.body
        return self.generic_visit(node)


def main():
    before=read(E/'before.json');snapshot=Path(before['archive'])/'source'
    changed=[];unchanged=[]
    for name,digest in before['files'].items():
        assert (ROOT/name).is_file(),name
        (unchanged if sha(ROOT/name)==digest else changed).append(name)
    expected={'tools/cdlno_perf/measure.py','README.md','tran_evaluate/msar_lno/README.md',
        'docs/MSAR_LNO_IMPLEMENTATION_STATUS.md','docs/CDLNO_IMPLEMENTATION_STATUS.md','memory/current-state.md'}
    assert set(changed)==expected,changed
    added=['tools/msar_benchmark.py','tools/cdlno_perf/msar.py','tests/test_msar_performance.py',
        'docs/MSAR_LNO_IMPLEMENTATION_REPORT.md','docs/MSAR_LNO_REQUIREMENTS_MATRIX.md',
        'docs/msar_lno_audit/m9/finalize_evidence.py']
    patch=[]
    for name in sorted(changed+added):
        prior=(snapshot/name).read_text() if name in before['files'] else ''
        current=(ROOT/name).read_text()
        patch.extend(difflib.unified_diff(prior.splitlines(True),current.splitlines(True),fromfile='before/'+name,tofile='after/'+name))
        if name.endswith('.py'):ast.parse(current,feature_version=(3,10))
    (E/'stage.patch').write_text(''.join(patch))
    new=OriginalTimingDefaults().visit(ast.parse((ROOT/'tools/cdlno_perf/measure.py').read_text()))
    old=ast.parse((snapshot/'tools/cdlno_perf/measure.py').read_text())
    assert ast.dump(new)==ast.dump(old),'old measurement default AST differs'
    old_readme=(snapshot/'README.md').read_text();new_readme=(ROOT/'README.md').read_text()
    added_readme=new_readme[new_readme.index('## 独立 MSAR-LNO（msar_lno）'):new_readme.index('---\n\n以下保留原 Transolver README')]
    assert new_readme.replace(added_readme,'',1)==old_readme,'prior README text changed'
    logs=[]
    for file,count,status in [('performance-tests.log',20,'OK'),
            ('independent-review-tests.log',37,'FAILED (errors=1)'),('command-test-correction.log',1,'OK')]:
        text=(E/file).read_text();match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',text)
        assert match and int(match[1])==count and '\n'+status in text,file
        logs.append(dict(log=file,methods=count,seconds=float(match[2]),reported_status=status))
    first=(E/'independent-review-tests.log').read_text();method='test_eight_task_four_variants_real_train_parsers_and_eval_previews'
    assert re.findall(r'^(?:FAIL|ERROR): (\w+) ',first,re.M)==[method]
    assert "TypeError: arguments() got an unexpected keyword argument 'flags'" in first
    assert re.search('^'+method+r'.*', (E/'command-test-correction.log').read_text(),re.M)
    inv=read(E/'parameter-cost-inventory.json')['inventory'];assert len(inv)==16
    for row in inv:
        p=row['parameters'];assert p['total']==sum(p['by_component'].values())
        assert p['by_component']['pair3']==3*row['architecture']['d'] and p['coverage_parameters']==0
        assert p['total']==p['trainable']
    rows=[];pairs=[]
    for file,size in [('gpu-n972.json',7),('gpu-air32000.json',4),('gpu-air32000-efficient.json',2)]:
        data=read(E/file);assert not data['errors'] and len(data['results'])==size
        settings=data['settings']
        assert settings['B']==1 and settings['precision']=='fp32' and not settings['tf32'] and not settings['compile']
        assert (settings['warmup'],settings['iterations'])==(5,20)
        for source,digest in data['source_sha256'].items():assert sha(ROOT/source)==digest,source
        grouped={}
        for row in data['results']:
            assert row['status']=='passed',row
            if row['model'].startswith('msar_'):
                key=row['model'].split('_')[1];grouped.setdefault(key,[]).append(row)
                assert row['cost']['verified_live']
                assert row['objective']['coverage_kappa']==.2 and row['objective']['coverage_weight']==.01
                assert len(row['training_path']['attention'])==(4 if row['model'].endswith('floor') else 0)
            measurement=row['measurement']
            assert measurement['optimizer']['successful_steps_per_active_parameter']==[25]
            for field in ('forward','train_step'):
                assert len(measurement[field]['samples_ms'])==20
                assert measurement[field]['peak_allocated_bytes']>0
            rows.append(dict(file=file,model=row['model'],N=data['case']['N'],backend=settings['backend'],
                forward=measurement['forward'],train_step=measurement['train_step']))
        for profile,pair in grouped.items():
            assert len(pair)==2 and pair[0]['initial_sha256']==pair[1]['initial_sha256']
            assert pair[1]['same_weight_eval_off_floor']==dict(passed=True,atol=0,rtol=0)
            off,on=[r['measurement']['train_step'] for r in pair]
            pairs.append(dict(file=file,profile=profile,inference_same_weight_exact=True,
                floor_step_median_change_percent=100*(on['median_ms']/off['median_ms']-1),
                floor_peak_allocated_change_MiB=(on['peak_allocated_bytes']-off['peak_allocated_bytes'])/2**20))
    assert len(rows)==13
    for row in read(E/'initial-audit.json')['results']:assert row['status']=='passed'
    index=read(ROOT/'docs/msar_lno_audit/m0/fixture-index.json')
    for item in index['integrity_files']:assert sha(item['path'])==item['sha256'],item['path']
    m8=read(ROOT/'docs/msar_lno_audit/m8/summary.json')
    assert m8['old_same_weight_exact']==75 and m8['old_hashes_preserved']==182
    # The numerical references must remain independent of production forwards.
    for path in ('tests/msar_reference.py','tests/msar_core_reference.py'):
        tree=ast.parse((ROOT/path).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith(('cdlno','tools'))
            if isinstance(node,ast.Import):assert not any(a.name.startswith(('cdlno','tools')) for a in node.names)
    write('freeze.json',dict(snapshot=str(snapshot),head=before['head'],branch=before['branch'],
        files_at_start=len(before['files']),changed=changed,added=added,unchanged_count=len(unchanged),
        unchanged_sha256={p:before['files'][p] for p in unchanged},current_sha256={p:sha(ROOT/p) for p in changed+added},
        measurement_old_default_AST='exact after projection of only explicit new callback',
        original_README_text='exact after removal of added MSAR section',
        production_models_tasks_data_config_dependencies_old_tests='byte-identical to pre-M9',
        inherited_M8_old_same_weight_replays=75,old_fixture_hashes_rechecked=182))
    write('summary.json',dict(stage='M9',environment=read(E/'gpu-n972.json')['environment'],
        suites=logs,new_methods=6,distinct_final_passed_methods=57,unresolved_failures=0,
        initial_new_test_error='wrong helper signature; corrected only new command test, original log preserved',
        actual_profile_parameter_models=16,live_CPU_audit_cases=2,GPU_measured_rows=13,
        GPU_settings='FP32 B1, TF32/AMP/compile off, warmup5 measured20, synchronized, initialized AdamW state',
        cases=rows,coverage_pairs=pairs,train_actual_parser_previews=32,eval_shell_previews=32,
        old_model_evidence='M8 75 exact same-weight replays retained, not falsely counted as M9 reruns; all182 artifact hashes rechecked',
        original_performance_test_methods=15,production_modified=False,real_data_training=False,
        limits=['remote Torch2.11cu128','real dataset read completeness','convergence','accuracy','real epoch duration','SOTA',
            'complete industrial sampling/radius_graph/VTK/forces','all default task batches/GPU AMP/compile matrix','complete optimizer/RNG resume'],
        next_stage_executed=False))
    for doc in ('docs/MSAR_LNO_IMPLEMENTATION_REPORT.md','docs/MSAR_LNO_REQUIREMENTS_MATRIX.md','tran_evaluate/msar_lno/README.md'):
        path=ROOT/doc
        markdown=path.read_text()
        prose=re.sub(r'```.*?```','',markdown,flags=re.S)
        prose=re.sub(r'(`+).*?\1','',prose)
        for link in re.findall(r'\]\(([^)]+)\)',prose):
            if '://' not in link and not link.startswith('#'):assert (path.parent/link.split('#')[0]).exists(),link
        for block in re.findall(r'```bash\n(.*?)```',markdown,re.S):
            subprocess.run(['bash','-n'],input=block,text=True,check=True)
    subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==before['head']
    (E/'status-after.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
    print(json.dumps(dict(stage='M9',changed=len(changed),added=len(added),unchanged=len(unchanged),
        parameter_models=16,GPU_rows=13,final_distinct_tests=57,old_fixture_hashes=182,production_changed=False)))


if __name__=='__main__':main()
