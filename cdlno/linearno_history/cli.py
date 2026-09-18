"""Orthogonal history flags, explicitly parsed without legacy default leakage."""
import argparse
from pathlib import Path
import json
from linearno_history.schema import FEATURE_FIELDS


def take_features(tokens):
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    for name in FEATURE_FIELDS[:2]:
        parser.add_argument('--'+name,'--'+name.replace('_','-'),choices=('0','1'),default=argparse.SUPPRESS)
    parser.add_argument('--'+FEATURE_FIELDS[2],'--'+FEATURE_FIELDS[2].replace('_','-'),type=float,default=argparse.SUPPRESS)
    namespace,remaining=parser.parse_known_args(tokens)
    explicit=vars(namespace)
    for key in FEATURE_FIELDS[:2]:
        if key in explicit:explicit[key]=explicit[key]=='1'
    return explicit,remaining


def run_family(tokens):
    probe=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    probe.add_argument('--experiment-dir','--linearno-run-dir',dest='directory',type=Path)
    args,_=probe.parse_known_args(tokens)
    path=args.directory/'architecture.json' if args.directory else None
    if path is not None and path.is_file():return json.loads(path.read_text()).get('family')
    return None
