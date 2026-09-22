"""Verify persisted previews with the independent shape oracle; no torch."""
import builtins
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'tests'), str(ROOT)]
old_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'numpy') or name.startswith(('cdlno.linearno_loop', 'tran_evaluate')):
        raise AssertionError('forbidden import: ' + name)
    return old_import(name, *args, **kwargs)


builtins.__import__ = guarded
from linearno_loop.v3.config import validate_config, run_directory_id
from loop_linearno_latent_adapter.oracle import count

started = time.perf_counter()
ids, hashes, tuples = set(), set(), set()
counts = Counter()
pairing, data_seeds = {}, {}
maximum_length = 0
records = 0
for line in (OUT / 'matrix/configuration-matrix.jsonl').open():
    row = json.loads(line)
    c = validate_config(row['config'])
    s, m = c['loop_spec'], c['profile_spec']['values']['model']
    cost = row['cost']
    task = s['task']
    seed = c['profile_spec']['values']['runtime']['seed']
    axis = (task, s['cost_profile'], s['executed_depth'], s['residual_mode'], s['latent_enabled'], s['adapter_mode'], seed)
    assert axis not in tuples
    tuples.add(axis)
    identifier = run_directory_id(c)
    assert identifier == row['preview']['run_id'] and identifier not in ids
    ids.add(identifier)
    assert c['config_hash'] not in hashes
    hashes.add(c['config_hash'])
    maximum_length = max(maximum_length, len(identifier.encode()))
    assert len(identifier.encode()) <= 255
    assert row['preview']['status'] == 'SCHEMA_PREVIEW_NOT_RUNNABLE_LAA1'
    assert (s['prefix_blocks'], s['suffix_blocks'], s['loop_repeats']) == (2, 2, 2)
    assert s['recurrent_core_blocks'] == (s['executed_depth'] - 4) // 2
    assert s['comparator_depth'] == s['unique_depth']
    assert s['hidden_width'] == m['hidden_width'] == c['resolution']['hidden_width'] == c['model_spec']['constructor_kwargs']['hidden_width']
    assert s['actual_M'] == (32 if task in ('ns', 'car', 'airfrans') else 64)
    for field, original in [('grid_height', 'H'), ('grid_width', 'W')]:
        assert s[field] == m[field] == c['model_spec']['constructor_kwargs'][field] == c['base_profile_spec']['values']['model'][original]
    pair = (task, s['cost_profile'], s['executed_depth'], seed)
    assert pairing.setdefault(pair, c['fair_comparison']) == c['fair_comparison']
    data_key = (task, seed)
    assert data_seeds.setdefault(data_key, c['fair_comparison']['dataloader_generators']) == c['fair_comparison']['dataloader_generators']
    channel = (m['ref'] ** 2 if m['unified_pos'] else m['space_dim']) + m['fun_dim']
    if task == 'airfrans' and m['unified_pos']:
        channel += m['space_dim']
    expected = count(task, s['hidden_width'], s['latent_width'], s['actual_M'], s['heads'],
                     2, s['recurrent_core_blocks'], 2, 2, m['ffn_ratio'], cost['batch'], cost['points'],
                     channel, m['out_dim'], s['variant'], m['time_input'], s['latent_enabled'],
                     s['adapter_mode'] != 'none', rank=s['adapter_rank'], residual=s['residual_mode'])
    assert cost['total_parameters'] == expected['parameters']
    assert cost['matrix_macs'] == expected['matrix_macs']
    assert cost['router_contraction_macs'] == expected['router_contraction_macs']
    assert cost['matrix_flops'] == 2 * cost['matrix_macs']
    assert cost['actual_parameters'] is None and cost['latency'] is None
    counts[task] += 1
    records += 1
assert records == 2304 and set(counts.values()) == {288}
custom = json.loads((OUT / 'matrix/custom-cases.json').read_text())
assert len(custom['valid']) == 3 and len(custom['invalid']) == 9
assert all(row['status'] == 'EXPECTED_REJECTION' for row in custom['invalid'])
result = dict(status='PASS', records=records, unique_run_ids=len(ids), unique_hashes=len(hashes),
              task_counts=dict(counts), paired_backbone_groups=len(pairing), data_seed_groups=len(data_seeds),
              maximum_run_id_bytes=maximum_length, independent_integer_parameter_max_error=0,
              independent_integer_mac_max_error=0, custom_valid=3, custom_expected_rejections=9,
              no_torch_import=True, no_model_construction=True, no_training=True,
              wall_seconds=time.perf_counter()-started,
              artifacts={str(p.relative_to(OUT)): dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                         for p in sorted((OUT / 'matrix').iterdir())})
with (OUT / 'preview-verification.json').open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps(result))
