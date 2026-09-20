"""Regenerate LL1 configuration/metadata fixtures; never constructs a model.

Run from repo: PYTHONPATH=tests:. PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
python -B docs/loop_linearno_audit/ll1/build_contract_artifacts.py
NumPy/torch are used ONLY by the synthetic numeric-state fixture producer.
"""
from pathlib import Path
import json

from linearno_loop.config import resolve_config
from linearno_loop.contracts import (
    ARCHITECTURE_EXTENSION, CLASS_PATHS, CLI_CONTRACT, CONFIG_VERSION, FAMILY,
    FORMULA_VERSION, PRESETS, RESIDUAL_MODES, SCHEMA_VERSION, SHARING, TOPOLOGY_FIELDS,
)
from linearno_loop.matrix import configuration_matrix
from linearno_loop.schema import LOAD_POLICY, PROVENANCE_FIELDS, REFERENCE_PINS, SECTIONS
from loop_linearno.support import metadata

OUT=Path(__file__).resolve().parent


def write(name,value,compact=False):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,
        indent=None if compact else 2,allow_nan=False)+'\n')


if __name__=='__main__':
    catalog=dict(format='loop-configuration-contract-catalog-v1',
        authoritative_validators=['linearno_loop.config.validate_config','linearno_loop.schema.validate_metadata'],
        note='Machine-readable contract catalog, not a replacement JSON Schema engine.',
        family=FAMILY,architecture_extension=ARCHITECTURE_EXTENSION,
        config_version=CONFIG_VERSION,schema_version=SCHEMA_VERSION,formula_version=FORMULA_VERSION,
        topology_presets={k:dict(zip(TOPOLOGY_FIELDS,v)) for k,v in PRESETS.items()},
        topology_rules=dict(custom='all four integer fields required; no bool/string/coercion',
            preset='no explicit P/C/R/S allowed',minimum=dict(zip(TOPOLOGY_FIELDS,(0,1,1,1))),
            unique_depth='P+C+S',executed_depth='P+C*R+S',derived_fields='read-only'),
        residual_modes=list(RESIDUAL_MODES),topology_default=None,residual_mode_default=None,
        rank=dict(default_multiplier=2,allowed_multipliers=[1,2],
            explicit_actual_policy='explicit_actual with rank_multiplier=null',
            conflict='explicit actual M and explicit multiplier always reject',
            shapenet='resolved_rank % head_dim == 0',base_source='unchanged current task/profile'),
        sharing=SHARING,cli=CLI_CONTRACT,cli_status='PLANNED_NOT_CONNECTED',
        class_paths=CLASS_PATHS,class_status='PLANNED_NOT_IMPLEMENTED_NO_PLACEHOLDERS',
        sections=list(SECTIONS),load_policy=LOAD_POLICY,provenance_fields=list(PROVENANCE_FIELDS),
        reference_pins=REFERENCE_PINS,unknown_fields='reject',nonfinite_json='reject',
        physical_time='native task T preserved; loop feature timestep encoding forbidden')
    write('contract-catalog.json',catalog)
    write('configuration-matrix.json',configuration_matrix(),compact=True)
    example=resolve_config('darcy',options={'topology_preset':'p1_c3_r2_s1','residual_mode':'rb_attnres'})
    write('synthetic-metadata.json',metadata(example),compact=True)
    cases=[
        ('strict-types','bool as int / string number / NaN / unknown fields','test_config'),
        ('topology','preset+custom / incomplete PCRS / P<0 or C,R,S<1','test_config'),
        ('derived','rehash cannot legitimize wrong depth/head_dim/rank/counts','test_config'),
        ('rank','explicit multiplier+actual M / ShapeNet noninteger multiple','test_config'),
        ('history-mixing','any old A/K/dropout field even false/None','test_config;test_schema'),
        ('protocol','objective/evaluation/data overrides rejected','test_config'),
        ('metadata','missing sections / class kwargs / family/schema/protocol mismatch','test_schema'),
        ('load','strict=False / baseline/history as loop / resume architecture changes','test_schema'),
        ('normalizer','missing fit/checksum/states / dtype/shape/bytes/hash corruption','test_schema'),
        ('resume','incomplete optimizer/scheduler/RNG/generator/sampler; test-selected best','test_schema'),
        ('ensemble','duplicate member / order / unsafe path / whole object','test_schema'),
        ('isolation','torch/model/task import / old source mutation / RNG advance','test_isolation'),
    ]
    write('negative-cases.json',{'schema_version':1,'cases':[
        dict(id=i,reject=c,test_module='tests/loop_linearno/'+t) for i,c,t in cases],
        'future_model_only_not_run':['tensor shapes and gradients','actual shared module identity',
            'forward-local history','state_dict strict loading','optimizer/RNG backend restore']})
    print('Wrote contract, 144 main + 144 rank-x1 + 24 custom previews, synthetic metadata, negative cases.')
