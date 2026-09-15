"""Read-only remote file/metadata inventory; never imports task loaders or torch.

NPY headers, MAT variable names/shapes, manifest lengths and bounded file search
help locate data. This does not validate all values, node order or normalization.
Prints JSON to stdout; redirect it to a report outside the dataset if desired.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import struct


def npy_header(stream):
    if stream.read(6) != b'\x93NUMPY':
        raise ValueError('not an NPY header')
    version = tuple(stream.read(2))
    if version == (1, 0):
        size = struct.unpack('<H', stream.read(2))[0]
    elif version in ((2, 0), (3, 0)):
        size = struct.unpack('<I', stream.read(4))[0]
    else:
        raise ValueError(f'unsupported NPY version {version}')
    if size > 1024 * 1024:
        raise ValueError('header exceeds 1MiB inspection limit')
    header = ast.literal_eval(stream.read(size).decode('utf-8' if version[0] == 3 else 'latin1'))
    return dict(format='npy', shape=list(header['shape']), dtype=header['descr'],
                fortran_order=header['fortran_order'], values_read=False)


def metadata(path):
    record = dict(path=str(path), exists=path.is_file())
    if not record['exists']:
        return record
    try:
        record['bytes'] = path.stat().st_size
        if path.suffix.lower() == '.npy':
            with path.open('rb') as stream:
                record.update(npy_header(stream))
        elif path.suffix.lower() in ('.mat', '.h5', '.hdf5'):
            with path.open('rb') as stream:
                header = stream.read(520)
            hdf5 = header[:8] == b'\x89HDF\r\n\x1a\n' or header[512:520] == b'\x89HDF\r\n\x1a\n'
            if hdf5:
                import h5py
                with h5py.File(path, 'r') as data:
                    record['variables'] = {key:dict(shape=list(value.shape),dtype=str(value.dtype))
                                           for key,value in data.items() if hasattr(value,'shape')}
                record['format'] = 'hdf5 / possibly MATLAB v7.3'
                record['entry_compatibility'] = 'Original scipy.io.loadmat does not read v7.3; do not transpose/convert silently.'
            else:
                from scipy.io import whosmat
                record['variables'] = {key:dict(shape=list(shape),dtype=kind) for key,shape,kind in whosmat(path)}
                record['format'] = 'MAT readable by scipy.io.whosmat'
            record['values_read'] = False
        elif path.name == 'manifest.json':
            if record['bytes'] > 8 * 1024 * 1024:
                raise ValueError('manifest exceeds 8MiB inspection limit')
            manifest = json.loads(path.read_text())
            if not isinstance(manifest, dict):
                raise ValueError('expected a split-name to sample-list mapping')
            record['splits'] = {key:dict(count=len(value),examples=value[:2])
                                for key,value in manifest.items() if isinstance(value,list)}
            record['format'] = 'json manifest'
            examples = next((value for value in manifest.values() if isinstance(value,list) and value), [])
            if examples and isinstance(examples[0], str):
                sample = examples[0]
                record['first_sample_files'] = {suffix:(path.parent/sample/(sample+suffix)).is_file()
                                                for suffix in ('_internal.vtu','_aerofoil.vtp')}
        elif path.suffix.lower() in ('.vtk', '.vtu', '.vtp'):
            with path.open('rb') as stream:
                prefix = stream.read(65536)
            prefix = prefix.split(b'<AppendedData',1)[0].decode('utf-8',errors='replace')
            record.update(format='bounded VTK header', prefix_limit_bytes=65536,
                          point_counts=re.findall(r'(?:NumberOfPoints="|POINTS\s+)(\d+)',prefix)[:4],
                          array_names=re.findall(r'<DataArray\b[^>]*\bName="([^"]+)"',prefix)[:30],
                          full_mesh_read=False)
        else:
            record['format'] = 'file present; contents not read'
    except Exception as error:
        record['metadata_error'] = f'{type(error).__name__}: {error}'
    return record


def expected_files(paths):
    f = lambda p, expected: dict(path=Path(p), expected=expected)
    return {
        'darcy': [f(paths['darcy']/f'piececonst_r421_N1024_smooth{i}.mat', 'coeff/sol [S,421,421]; train1000/test200') for i in (1,2)],
        'elasticity': [f(paths['elasticity']/'elasticity/Meshes'/name, shape) for name,shape in (
            ('Random_UnitCell_XY_10.npy','[972,2,S], S>=1200'),
            ('Random_UnitCell_sigma_10.npy','[972,S], S>=1200'))],
        'airfoil': [f(paths['airfoil']/f'NACA_Cylinder_{key}.npy',shape) for key,shape in (
            ('X','[S,221,51], S>=1200'),('Y','[S,221,51], S>=1200'),('Q','[S,C,221,51], C>=5; use channel4'))],
        'pipe': [f(paths['pipe']/f'Pipe_{key}.npy',shape) for key,shape in (
            ('X','[S,129,129], S>=1200'),('Y','[S,129,129], S>=1200'),('Q','[S,C,129,129], C>=1; use channel0'))],
        'ns': [f(paths['ns']/'NavierStokes_V1e-5_N1200_T20/NavierStokes_V1e-5_N1200_T20.mat',
                 'u [S,64,64,20], S>=1200; viscosity1e-5; 10->10')],
        'plasticity': [f(paths['plasticity'],'input [S,101], output [S,101,31,20,4]; official S987, use900/80')],
        'airfrans': [f(paths['airfrans']/'manifest.json','full/scarce/reynolds/aoa split lists; sample/sample_internal.vtu and sample_aerofoil.vtp')],
    }


def car_inventory(raw, cache):
    folds = []
    required = ('x.npy','y.npy','pos.npy','surf.npy','edge_index.npy')
    example_files = []
    for i in range(9):
        folder = raw/f'param{i}'
        row = dict(fold=i, raw_directory_exists=folder.is_dir())
        if folder.is_dir():
            # Bounded inspection; never alter the original loader's ordering.
            samples=[]
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_dir(): samples.append(entry.name)
                    if len(samples) >= 2000: break
            row['sample_directories_observed']=len(samples)
            row['count_may_be_truncated']=len(samples)>=2000
            absent_raw=[]; absent_cache=[]
            for name in samples:
                sample=Path(f'param{i}')/name
                if not all((raw/sample/file).is_file() for file in ('quadpress_smpl.vtk','hexvelo_smpl.vtk')):
                    absent_raw.append(name)
                if not all((cache/sample/file).is_file() for file in required):
                    absent_cache.append(name)
            row.update(incomplete_raw_count=len(absent_raw),incomplete_cache_count=len(absent_cache),
                       incomplete_raw_examples=absent_raw[:5],incomplete_cache_examples=absent_cache[:5])
            if samples and not example_files:
                sample=Path(f'param{i}')/samples[0]
                example_files=[metadata(cache/sample/file) for file in required]
                example_files += [metadata(raw/sample/file) for file in ('quadpress_smpl.vtk','hexvelo_smpl.vtk')]
        folds.append(row)
    canonical=Path('/data/PDE_data/mlcfd_data/training_data')
    return dict(folds=folds,representative_headers=example_files,
                expected='cache x[N,7], y[N,4], pos[N,3], surf[N], edge_index[2,E]; variableN allowed',
                note='Original loader skips missing raw/cache samples. Counts/file presence do not prove label/node order correctness.',
                drag_evaluation=dict(required_fold=0,hardcoded_root=str(canonical),
                                     canonical_exists=canonical.is_dir(),same_raw_root=canonical.resolve()==raw.resolve()))


def discover(root, max_depth, max_entries):
    markers={'manifest.json','Random_UnitCell_XY_10.npy','NACA_Cylinder_X.npy','Pipe_X.npy',
             'plas_N987_T20.mat','piececonst_r421_N1024_smooth1.mat','NavierStokes_V1e-5_N1200_T20.mat',
             'quadpress_smpl.vtk','x.npy'}
    hits={}; errors=[]; visited=0; truncated=False; archives=[]
    for directory,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
        relative=Path(directory).relative_to(root)
        dirs[:]=sorted(d for d in dirs if not d.startswith('.'))
        if len(relative.parts)>=max_depth: dirs[:]=[]
        visited+=len(dirs)+len(files)
        for name in sorted(files):
            path=str(Path(directory)/name)
            if name in markers:
                bucket=hits.setdefault(name,[])
                if len(bucket)<4: bucket.append(path)
            elif name.endswith(('.zip','.tar','.tar.gz','.tgz','.7z')) and len(archives)<20:
                archives.append(path)
        if visited>=max_entries:
            truncated=True
            break
    return dict(candidates=hits,archives_not_extracted=archives,entries_observed=visited,
                truncated=truncated,max_depth=max_depth,errors=errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path(os.environ.get('CDLNO_DATA_ROOT','.')))
    parser.add_argument('--max-depth',type=int,default=6)
    parser.add_argument('--max-entries',type=int,default=20000)
    parser.add_argument('--no-search',action='store_true',help='Only inspect configured expected paths')
    args=parser.parse_args()
    if args.max_depth<0 or args.max_entries<1: parser.error('require max-depth>=0 and max-entries>=1')
    keys=dict(darcy='CDLNO_DARCY_ROOT',elasticity='CDLNO_ELASTICITY_ROOT',airfoil='CDLNO_AIRFOIL_ROOT',
              pipe='CDLNO_PIPE_ROOT',ns='CDLNO_NS_ROOT',plasticity='CDLNO_PLASTICITY_FILE',airfrans='CDLNO_AIRFRANS_DATASET')
    defaults=dict(darcy='fno',elasticity='fno',airfoil='fno/airfoil/naca',pipe='fno/pipe',ns='fno',
                  plasticity='fno/plas_N987_T20.mat',airfrans='AirfRANS/Dataset')
    # --data-root controls discovery only; path.sh's per-task settings remain explicit.
    paths={key:Path(os.environ.get(env,str(args.data_root/defaults[key]))) for key,env in keys.items()}
    report=dict(scope='read-only metadata; no dataset loaders/training/normalization/value or node-order validation',
                search_root=str(args.data_root),root_exists=args.data_root.is_dir(),
                configured_paths={k:str(v) for k,v in paths.items()},tasks={})
    for task,files in expected_files(paths).items():
        report['tasks'][task]=[dict(expected=item['expected'],**metadata(item['path'])) for item in files]
    raw=Path(os.environ.get('CDLNO_CAR_RAW_ROOT',str(args.data_root/'mlcfd_data/training_data')))
    cache=Path(os.environ.get('CDLNO_CAR_CACHE_ROOT',str(args.data_root/'mlcfd_data/preprocessed_data')))
    try: report['tasks']['car']=dict(raw_root=str(raw),cache_root=str(cache),**car_inventory(raw,cache))
    except OSError as error: report['tasks']['car']={'inspection_error':str(error)}
    if not args.no_search:
        report['discovery']=discover(args.data_root,args.max_depth,args.max_entries)
        # A few marker headers reveal formats even when configured paths are wrong.
        report['candidate_headers']=[metadata(Path(items[0])) for key,items in report['discovery']['candidates'].items()
                                     if key not in ('quadpress_smpl.vtk','x.npy')]
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report['root_exists'] else 2


if __name__=='__main__':
    raise SystemExit(main())


