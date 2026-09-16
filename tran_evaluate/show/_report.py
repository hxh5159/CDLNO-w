#!/usr/bin/env python3
"""Read-only experiment postprocessing. No torch, models, datasets or pickle loads."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import uuid
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

PROJECTS = {
    'Car-Design-ShapeNetCar': ('car',),
    'Airfoil-Design-AirfRANS': ('airfrans',),
    'PDE-Solving-StandardBenchmark': ('darcy', 'airfoil', 'plasticity', 'elasticity', 'ns', 'pipe'),
}
ALIASES = {'elas':'elasticity','plas':'plasticity','shapenet-car':'car','shapenet_car':'car','navier-stokes':'ns'}
TITLES = {'ns':'Navier–Stokes','car':'ShapeNet-Car','airfrans':'AirfRANS'}
VALIDATION_KEYS = {
    **{task:('validation_relative_l2',) for task in ('darcy','airfoil','elasticity','pipe')},
    **{task:('test_full_loss',) for task in ('ns','plasticity')},
    'car':('validation_velocity_mse','validation_pressure_mse'),
    'airfrans':('validation_volume_mse','validation_surface_mse'),
}
SELECTION_POLICY = (
    'One run per task, minimizing the last recorded validation loss, never the historical minimum '
    'or independent evaluation scores. Known unfinished/failed runs are excluded. Legacy runs '
    'without a training status file require a recorded validation loss and are marked unverified. '
    'Multiple members within one run use the arithmetic mean of their respective final validation '
    'losses, without selecting a member. Exact ties use recorded integer seed (ascending, missing '
    'last), then absolute path (lexicographic). No source run or previous report is modified.'
)
STYLE = {'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':8,
         'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
         'legend.fontsize':7,'axes.linewidth':.6,'lines.linewidth':1.25,
         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
         'savefig.facecolor':'white','figure.facecolor':'white',
         'axes.prop_cycle':matplotlib.cycler(color=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9'])}
NUM = r'(?:[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?|[-+]?(?i:nan|inf(?:inity)?))'


def slug(value):
    return re.sub(r'[^\w.-]+','_',str(value),flags=re.ASCII).strip('._')[:100] or 'unnamed'


def flatten(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():yield from flatten(item, f'{prefix}.{key}' if prefix else key)
    elif isinstance(value, (list, tuple)):
        for i,item in enumerate(value):yield from flatten(item,f'{prefix}[{i}]')
    else:yield prefix,value


def number(value):
    if isinstance(value,bool) or value is None:return math.nan
    try:return float(value)
    except (TypeError, ValueError):return math.nan


def write_json(path, value):
    def clean(item):
        if isinstance(item,dict):return {k:clean(v) for k,v in item.items()}
        if isinstance(item,(list,tuple)):return [clean(v) for v in item]
        if isinstance(item,float) and not math.isfinite(item):return str(item)
        return item
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def tex(text):
    mapping={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','$':r'\$','{':r'\{','}':r'\}'}
    return ''.join(mapping.get(c,c) for c in text)


def validation_value(task, metrics, config):
    """Saved held-out monitor, with named industrial components (not swapped log values)."""
    keys=VALIDATION_KEYS[task]
    values=[number(metrics.get(k)) for k in keys]
    if not all(math.isfinite(v) and v>=0 for v in values):
        raise ValueError('last validation has missing, nonfinite or negative components: '+', '.join(keys))
    weight=None
    if len(keys)==1:return keys[0],values[0],weight
    args=config.get('resolved_arguments',{})
    recorded=[number(v) for v in (metrics.get('reg'),args.get('weight')) if v is not None]
    if not recorded or not all(math.isfinite(v) and v>=0 for v in recorded):
        raise ValueError('industrial validation requires a saved finite nonnegative weight/reg')
    if any(v!=recorded[0] for v in recorded):raise ValueError('saved reg and weight disagree')
    if task=='airfrans' and metrics.get('criterion','MSE_weighted')!='MSE_weighted':
        raise ValueError('unsupported saved AirfRANS criterion; weighted MSE cannot be inferred')
    weight=recorded[0];value=values[0]+weight*values[1]
    if not math.isfinite(value):raise ValueError('nonfinite combined validation loss')
    return keys[0]+' + weight * '+keys[1],value,weight


class Bundle:
    def __init__(self, args, directory):
        self.args,self.directory=args,directory
        self.warnings=[];self.sources={};self.runs=[];self.eval_rows=[];self.epoch_rows=[]
        self.histories={};self.selection=[];self.history_errors=set()

    def warn(self, message):
        self.warnings.append(str(message));print('NOTE:',message,file=sys.stderr)

    def track(self,path,raw=None):
        if raw is None:raw=path.read_bytes()
        digest=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        if str(path) in self.sources and self.sources[str(path)]!=digest:
            self.warn(f'{path}: source changed during report generation; this is not an atomic snapshot of an active run')
        self.sources[str(path)]=digest
        return raw

    def json(self,path,default=None):
        if not path.is_file():return default
        try:return json.loads(self.track(path))
        except (ValueError,OSError) as e:self.warn(f'{path}: {e}');return default

    def save_figure(self, fig, path, caption):
        path.parent.mkdir(parents=True,exist_ok=True)
        try:
            for ext in ('pdf','png'):fig.savefig(path.with_suffix('.'+ext),dpi=self.args.dpi,bbox_inches='tight',pad_inches=.04)
        finally:plt.close(fig)
        path.with_suffix('.txt').write_text(caption+'\n',encoding='utf-8')
        path.with_suffix('.tex').write_text('\\caption{'+tex(caption)+'}\n',encoding='utf-8')

    def curve(self,path,series,title,ylabel,context):
        # Each series carries real recorded epoch coordinates, including sparse validation.
        usable=[(label,np.asarray(x),np.asarray(y,dtype=float)) for label,x,y in series if len(x)]
        if not usable:return
        fig,ax=plt.subplots(figsize=(3.5,2.65),layout='constrained')
        vals=np.concatenate([y[np.isfinite(y)] for _,_,y in usable])
        logarithmic=len(vals)>0 and np.all(vals>0)
        for label,x,y in usable:
            ax.plot(x,np.where(np.isfinite(y),y,np.nan),label=label,
                    marker='o' if len(x)<30 else None,markersize=2)
        if logarithmic:ax.set_yscale('log')
        ax.set(xlabel='Completed epoch',ylabel=ylabel,title=title)
        ax.grid(which='major',alpha=.18);ax.legend(frameon=False,loc='best')
        self.save_figure(fig,path,context+' '+title+'. Original recorded values, without smoothing or resampling. '
                         'Validation points use their recorded epochs; connecting lines are visual guides. '
                         +('Logarithmic' if logarithmic else 'Linear')+' vertical axis. Missing/nonfinite values are not replaced by zero.')

    def history(self,run,task):
        # Reuse the exact snapshot used for selection when exporting winning curves.
        if (run,task) in self.histories:return self.histories[run,task]
        path=run/'train_history.jsonl';rows=[]
        if path.exists():
            try:lines=self.track(path).decode('utf-8').splitlines()
            except (OSError,UnicodeError) as e:self.history_errors.add((run,task));self.warn(f'{path}: {e}');return []
            for index,line in enumerate(lines):
                try:
                    row=json.loads(line)
                    if type(row.get('epoch')) is not int or row['epoch']<1 or not isinstance(row.get('metrics'),dict):raise ValueError('expected completed epoch and metrics')
                    row['member']=row.get('member',0);rows.append(row)
                except (ValueError,TypeError,AttributeError) as e:
                    self.history_errors.add((run,task));self.warn(f'{path}:{index+1}: ignored invalid/incomplete row ({e})')
        elif (run/'train.log').is_file() and task in PROJECTS['PDE-Solving-StandardBenchmark']:
            # Exact existing PDE print formats only; never infer epochs from tqdm.
            text=self.track(run/'train.log').decode('utf-8',errors='replace');current=None
            for line in text.splitlines():
                match=re.search(r'\bEpoch\s+(\d+)\s',line)
                if match:
                    metrics={}
                    for key in ('train_step_loss','train_full_loss','test_step_loss','test_full_loss'):
                        m=re.search(key+r'\s*:\s*('+NUM+')',line)
                        if m:metrics[key]=float(m.group(1))
                    m=re.search(r'Train loss\s*:\s*('+NUM+')',line)
                    if m:metrics['train_loss']=float(m.group(1))
                    m=re.search(r'\bReg\s*:\s*('+NUM+')',line)
                    if m:metrics['derivative_regularizer']=float(m.group(1))
                    current=dict(epoch=int(match.group(1))+1,member=0,metrics=metrics)
                    if metrics:rows.append(current)
                elif current is not None:
                    m=re.search(r'rel_err\s*:\s*('+NUM+')',line)
                    if m:current['metrics']['validation_relative_l2']=float(m.group(1));current=None
            if rows:self.warn(f'{run}: history recovered from rounded console text; JSONL was absent')
        seen={}
        for row in rows:
            key=(str(row['member']),row['epoch'])
            if key in seen:self.warn(f'{run}: duplicate member/epoch {key}; last recorded row used')
            seen[key]=row
        return sorted(seen.values(),key=lambda r:(str(r['member']),r['epoch']))

    def selection_record(self,run,task,config,architecture):
        args=config.get('resolved_arguments',{});arch=config.get('architecture') or architecture.get('architecture',{})
        result=self.json(run/'train_results.json',{})
        status=result.get('status','unrecorded') if isinstance(result,dict) else 'invalid'
        record=dict(source=str(run),task=task,model=config.get('model') or arch.get('family') or arch.get('model_name') or 'unknown',
                    seed=args.get('seed'),fold=args.get('fold_id'),split=args.get('task'),training_status=status,
                    selected=False,eligible=False,metric=None,final_validation_loss=None,validation_epochs={},
                    member_losses={},weights={},last_training_epochs={},reason='')
        if (run/'train_results.json').exists() and status!='completed':
            record['reason']='training status is not completed: '+str(status);return record
        rows=self.history(run,task);self.histories[run,task]=rows
        if (run,task) in self.history_errors:
            record['reason']='invalid/incomplete history rows; final validation cannot be confirmed';return record
        members={str(r['member']) for r in rows}
        members.update(str(m) for m in config.get('members',{}))
        count=args.get('nmodel')
        if type(count) is int and count>0:members.update(str(i) for i in range(count))
        if not members:record['reason']='no recorded validation history';return record
        for member in sorted(members):
            member_rows=[r for r in rows if str(r['member'])==member]
            record['last_training_epochs'][member]=max((r['epoch'] for r in member_rows),default=None)
            # A partial/nonfinite last validation must not fall back to an earlier finite one.
            validation=[r for r in member_rows if any(
                k.startswith('validation_') or k.startswith('test_') or k=='upstream_validation_log_value'
                for k in r['metrics'])]
            if not validation:record['reason']=f'member {member}: no recorded validation';return record
            last=validation[-1];record['validation_epochs'][member]=last['epoch']
            try:metric,value,weight=validation_value(task,last['metrics'],config)
            except ValueError as e:record['reason']=f'member {member}, epoch {last["epoch"]}: {e}';return record
            record['metric']=metric;record['member_losses'][member]=value
            if weight is not None:record['weights'][member]=weight
        record['final_validation_loss']=math.fsum(v/len(members) for v in record['member_losses'].values())
        record['eligible']=True;record['reason']='eligible'
        if status=='unrecorded':self.warn(f'{run}: no training status file; selecting from last recorded validation, completion unverified')
        return record

    def select_runs(self,candidates):
        self.selection=[self.selection_record(*item) for item in candidates]
        chosen=set()
        for task in PROJECTS[self.args.project]:
            eligible=[r for r in self.selection if r['task']==task and r['eligible']]
            if not eligible:continue
            def order(r):
                seed=r['seed'];seed_key=(0,seed) if type(seed) is int else (1,0)
                return r['final_validation_loss'],seed_key,r['source']
            winner=min(eligible,key=order);winner['selected']=True;chosen.add(winner['source'])
            for r in eligible:
                r['reason']='minimum final validation loss' if r is winner else 'larger final validation loss'
                if r is not winner and r['final_validation_loss']==winner['final_validation_loss']:
                    r['reason']='equal final validation loss; deterministic seed/path tie-break'
            # Still choose exactly one per task; expose mixed settings instead of claiming a seed-controlled comparison.
            signatures=set()
            for run,t,cfg,sidecar in candidates:
                if str(run) not in {r['source'] for r in eligible}:continue
                a=cfg.get('resolved_arguments',{})
                training={k:v for k,v in a.items() if k in ('epochs','lr','batch_size','weight','weight_decay','ntrain','downsample','nmodel','fold_id','task')}
                signatures.add(json.dumps(dict(model=cfg.get('model'),architecture=cfg.get('architecture') or sidecar.get('architecture'),
                                               training=training,hparams=cfg.get('hparams')),sort_keys=True))
            if len(signatures)>1:
                self.warn(f'{task}: candidate architectures/training settings/folds/splits differ; the winner is only best among included runs, not a controlled seed comparison. Narrow --model/--run-dir/--runs-root if needed.')
        for r in self.selection:
            if not r['eligible']:self.warn(f'{r["source"]}: excluded from figures: {r["reason"]}')
        return [item for item in candidates if str(item[0]) in chosen]

    def pde_overview(self):
        """One six-panel figure, one selected run per benchmark; never superpose seeds."""
        if not self.runs:return
        fig,axes=plt.subplots(2,3,figsize=(7.16,4.9),layout='constrained')
        captions=[]
        for i,(task,ax) in enumerate(zip(PROJECTS['PDE-Solving-StandardBenchmark'],axes.flat)):
            info=next((r for r in self.runs if r['task']==task),None)
            title=f'({chr(97+i)}) '+TITLES.get(task,task.title())
            if info is None:
                ax.set_title(title);ax.text(.5,.5,'No eligible saved run',ha='center',va='center',transform=ax.transAxes,fontsize=8)
                ax.set_axis_off();captions.append(title+': no eligible saved run.');continue
            selection=info['selection'];rows=self.history(Path(info['source']),task)
            members=sorted({str(r['member']) for r in rows});all_values=[]
            # Plasticity records a per-step train loss but a full-field held-out loss:
            # label them explicitly instead of treating them as the same quantity.
            keys=[('train_loss','Train field'),('validation_relative_l2','Held-out field')]
            if task=='ns':keys=[('train_full_loss','Train (teacher forced)'),('test_full_loss','Held-out (rollout)')]
            elif task=='plasticity':keys=[('train_step_loss','Train per time'),('test_full_loss','Held-out full field')]
            for member in members:
                mr=[r for r in rows if str(r['member'])==member]
                for (key,label),color in zip(keys,('#0072B2','#D55E00')):
                    points=[(r['epoch'],number(r['metrics'][key])) for r in mr if key in r['metrics']]
                    if not points:continue
                    x,y=zip(*points);y=np.asarray(y);all_values.extend(y[np.isfinite(y)])
                    ax.plot(x,np.where(np.isfinite(y),y,np.nan),color=color,label=label+(f' / member {member}' if len(members)>1 else ''),
                            marker='o' if len(x)<30 else None,markersize=2)
            if all_values and np.all(np.asarray(all_values)>0):ax.set_yscale('log')
            seed=info['seed'] if info['seed'] is not None else 'unrecorded'
            epochs='/'.join(str(e) for e in sorted(set(selection['validation_epochs'].values())))
            ax.set(title=title+f' · seed {seed}',xlabel='Completed epoch',ylabel=r'Relative $L_2$')
            ax.text(.97,.97,f'Final held-out: {selection["final_validation_loss"]:.4g}\nValidation epoch: {epochs}',
                    transform=ax.transAxes,ha='right',va='top',fontsize=7,
                    bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=2))
            ax.grid(which='major',alpha=.18)
            if ax.lines:ax.legend(frameon=False,loc='upper right',bbox_to_anchor=(1,.81),fontsize=7)
            captions.append(title+': '+info['label']+f'; selected using {selection["metric"]}={selection["final_validation_loss"]:.6g} at epoch(s) {epochs}.')
        self.save_figure(fig,self.directory/'pde_selected_training',
                         'Six PDE benchmarks, one run per task chosen by lowest final recorded held-out loss. '
                         'Curves show original recorded values without smoothing; positive panels use a log vertical axis. '
                         'NS train uses truth-fed windows and held-out uses prediction-fed rollout; Plasticity train is per time, held-out is full field. '
                         'These are training-time held-out monitors (upstream test_* for NS/Plasticity), not fresh independent evaluations. '
                         +' '.join(captions))

    def pde_evaluation_overview(self):
        if not self.runs:return
        fig,axes=plt.subplots(2,3,figsize=(7.16,4.7),layout='constrained');captions=[]
        for i,(task,ax) in enumerate(zip(PROJECTS['PDE-Solving-StandardBenchmark'],axes.flat)):
            info=next((r for r in self.runs if r['task']==task),None)
            title=f'({chr(97+i)}) '+TITLES.get(task,task.title())
            rows=[r for r in self.eval_rows if r['task']==task and r['status']=='completed']
            keys=['test_step_loss','test_full_loss'] if task in ('ns','plasticity') else ['relative_l2']
            values=[r for r in rows if r['metric'] in keys and math.isfinite(number(r['value']))]
            seed=info['seed'] if info and info['seed'] is not None else 'unrecorded'
            ax.set_title(title+(f' · seed {seed}' if info else ''))
            if not values:
                ax.text(.5,.5,'No completed saved evaluation',ha='center',va='center',transform=ax.transAxes,fontsize=7)
                ax.set_axis_off();continue
            evaluations=list(dict.fromkeys(r['evaluation'] for r in values));x=np.arange(len(evaluations))
            present=[k for k in keys if any(r['metric']==k for r in values)];width=.65/len(present)
            for j,key in enumerate(present):
                y=[next((number(r['value']) for r in values if r['evaluation']==e and r['metric']==key),np.nan) for e in evaluations]
                bars=ax.bar(x+(j-(len(present)-1)/2)*width,y,width=width,
                            label={'relative_l2':'Field','test_step_loss':'Per step','test_full_loss':'Full field'}[key])
                for bar,value in zip(bars,y):
                    if math.isfinite(value):ax.annotate(f'{value:.3g}',(bar.get_x()+bar.get_width()/2,value),
                                                       xytext=(0,3),textcoords='offset points',ha='center',fontsize=7)
            ax.set_xticks(x,[f'E{j+1}' for j in range(len(evaluations))]);ax.set(xlabel='Saved evaluation',ylabel=r'Relative $L_2$')
            ax.set_xlim(-.75,len(evaluations)-.25)
            ax.margins(y=.22);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.legend(frameon=False,fontsize=7)
            captions.append(title+': '+info['label']+'; '+', '.join(f'E{j+1}={e}' for j,e in enumerate(evaluations))+'.')
        self.save_figure(fig,self.directory/'pde_selected_evaluation',
                         'Saved independent evaluation metrics for the same selected run in each of the six PDE tasks. '
                         'All completed saved evaluation attempts are shown; they are not used for seed selection. '
                         'Missing/failed evaluation results are not replaced with training validation or zeros. '+' '.join(captions))

    def training(self,run,dest,task,info,config):
        rows=self.history(run,task);long=[]
        for row in rows:
            for key,value in flatten(row['metrics']):
                item=dict(run=info['id'],task=task,model=info['model'],seed=info['seed'],member=row['member'],epoch=row['epoch'],metric=key,value=value)
                long.append(item)
        self.epoch_rows.extend(long)
        write_csv(dest/'training_history.csv',long,['run','task','model','seed','member','epoch','metric','value'])
        if not rows:
            self.warn(f'{run}: no numeric epoch history; no loss curve can be reconstructed from final weights/images')
            return
        for row in long:
            if isinstance(row['value'],(int,float)) and not math.isfinite(number(row['value'])):
                self.warn(f'{run}: nonfinite history {row["metric"]} at epoch {row["epoch"]}; gap retained')
        for member in sorted({str(r['member']) for r in rows}):
            member_rows=[r for r in rows if str(r['member'])==member]
            flat=[dict(flatten(r['metrics'])) for r in member_rows]
            path=dest/'curves'/f'member_{slug(member)}'
            context=info['label']+f'; member {member}.'
            def plot(name,keys,title,ylabel):
                series=[]
                for key,label in keys:
                    points=[(row['epoch'],number(metric[key])) for row,metric in zip(member_rows,flat) if key in metric]
                    if points:series.append((label,[p[0] for p in points],[p[1] for p in points]))
                self.curve(path/name,series,title,ylabel,context)
            if task in ('ns','plasticity'):
                plot('step_error',[('train_step_loss','Train'),('test_step_loss','Held-out')],'Per-step field error',r'Relative $L_2$')
                plot('trajectory_error',[('train_full_loss','Train (teacher forced)'),('test_full_loss','Held-out')],'Full-trajectory field error',r'Relative $L_2$')
            elif task=='car':
                plot('pressure_mse',[('train_components.pressure_mse','Train'),('validation_pressure_mse','Validation')],'Surface pressure', 'MSE (normalized)')
                plot('velocity_mse',[('train_components.velocity_mse','Train: all nodes'),('validation_velocity_mse','Validation: all nodes')],'Velocity', 'MSE (normalized)')
                # The old outer log swaps component order: retain it in CSV only.
                weight=number(config.get('resolved_arguments',{}).get('weight'))
                if math.isfinite(weight):
                    pts=[(r['epoch'],m['train_components.velocity_mse']+weight*m['train_components.pressure_mse']) for r,m in zip(member_rows,flat) if all(isinstance(m.get(k),(int,float)) for k in ('train_components.velocity_mse','train_components.pressure_mse'))]
                    self.curve(path/'training_objective',[('Train',[x for x,y in pts],[y for x,y in pts])],'Training objective','Velocity MSE + weight × pressure MSE',context+' Derived from saved correctly named components and saved weight; not upstream swapped log summary.')
            elif task=='airfrans':
                plot('training_objective',[('train_loss','Train')],'Training objective','Recorded weighted loss')
                for region in ('surface','volume'):
                    plot(region+'_mse',[(f'train_{region}_mse','Train'),(f'validation_{region}_mse','Validation')],region.title()+' fields','MSE (normalized)')
                    for i,ch in enumerate(('vx','vy','p','nut')):
                        plot(region+'_'+ch,[(f'train_{region}_per_channel[{i}]','Train'),(f'validation_{region}_per_channel[{i}]','Validation')],region.title()+' / '+ch,'MSE (normalized)')
            else:
                plot('field_error',[('train_loss','Train'),('validation_relative_l2','Held-out')],'Field prediction error',r'Relative $L_2$')
                if task=='darcy':plot('gradient_regularizer',[('derivative_regularizer','Unweighted gradient term')],'Darcy gradient regularizer','Recorded derivative loss')
        if task=='darcy':
            self.warn(f'{run}: train_loss is field relative L2; derivative_regularizer is unweighted, not the total backward objective')

    def evaluations(self,run):
        records=[]
        for f in sorted((run/'evaluations').glob('*/results.json')):
            data=self.json(f)
            if isinstance(data,dict):records.append((f.parent.name,f.parent,data))
        if not records:
            index=self.json(run/'eval_results.json',{})
            for i,data in enumerate(index.get('evaluations',[]) if isinstance(index,dict) else []):
                relative=data.get('result_file','');directory=(run/relative).parent.resolve()
                if not directory.is_relative_to(run.resolve()):
                    self.warn(f'{run}: ignored evaluation path outside the run');continue
                records.append((f'index_{i:03d}',directory,data))
        # Legacy explicit score roots and older automatic post-training Air scores.
        for directory in (run,run/'training_artifacts'):
            if (directory/'score.json').is_file() and all(d!=directory for _,d,_ in records):
                records.append(('saved_scores' if directory==run else 'training_artifacts',directory,
                                dict(status='legacy_unverified',metrics={'score':self.json(directory/'score.json',{})})))
        return records

    def bars(self,path,values,labels,title,ylabel,caption,std=None,correlation=False):
        values=np.asarray(values,dtype=float)
        if values.ndim!=1 or len(values)!=len(labels):raise ValueError('bar axes do not match')
        if not np.isfinite(values).any():return
        fig,ax=plt.subplots(figsize=(3.5,2.65),layout='constrained');x=np.arange(len(labels))
        ax.bar(x,values,color='#0072B2',width=.58)
        if std is not None:
            std=np.asarray(std,dtype=float)
            if std.shape!=values.shape or np.any(std<0):raise ValueError('invalid saved std shape/value')
            ax.errorbar(x,values,yerr=std,fmt='none',color='#333333',capsize=3,lw=.8)
        ax.set_xticks(x,labels);ax.set(title=title,ylabel=ylabel);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        if correlation:ax.set_ylim(-1.05,1.05)
        for xx,y in zip(x,values):
            if np.isfinite(y):ax.annotate(f'{y:.3g}',(xx,y),xytext=(0,4 if y>=0 else -10),textcoords='offset points',ha='center',fontsize=7)
        self.save_figure(fig,path,caption)

    def test_results(self,run,dest,task,info):
        evaluations=self.evaluations(run)
        if not evaluations:self.warn(f'{run}: no saved independent evaluation; training validation is not substituted for test results')
        for evaluation,directory,result in evaluations:
            target=dest/'evaluation'/slug(evaluation);target.mkdir(parents=True)
            metrics=result.get('metrics',{});score=self.json(directory/'score.json')
            if isinstance(score,dict):metrics=dict(metrics,score=score)
            write_json(target/'saved_results.json',result)
            rows=[dict(run=info['id'],task=task,model=info['model'],seed=info['seed'],evaluation=evaluation,status=result.get('status','unknown'),metric=key,value=value) for key,value in flatten(metrics)]
            self.eval_rows.extend(rows)
            write_csv(target/'metrics.csv',rows,['run','task','model','seed','evaluation','status','metric','value'])
            # Numeric tables are also directly reusable in a manuscript.
            table=['\\begin{tabular}{lr}','\\hline','Metric & Value \\\\','\\hline']
            table += [tex(r['metric'])+' & '+format(number(r['value']),'.6g')+r' \\' for r in rows if math.isfinite(number(r['value']))]
            (target/'metrics.tex').write_text('\n'.join(table+['\\hline','\\end{tabular}'])+'\n')
            status=result.get('status','unknown')
            caption=info['label']+f'; evaluation {evaluation}; recorded status {status}. Values are from saved evaluation, not recomputed from training loss.'
            if status!='completed':
                self.warn(f'{run}: {evaluation} status={status}; partial/legacy metrics retained in tables, not displayed as completed evaluation bars')
            else:
                groups= [('relative_l2',['relative_l2'],['Field'],r'Relative $L_2$'),('trajectory',['test_step_loss','test_full_loss'],['Per step','Full trajectory'],r'Relative $L_2$')]
                if task=='car':groups=[('relative_l2',['relative_l2_pressure','relative_l2_velocity'],['Pressure','Velocity'],r'Relative $L_2$'),('rmse',['rmse_pressure','rmse_velocity'],['Pressure','Velocity'],'RMSE (normalized)'),('drag_error',['c_d'],['Drag'],'Mean relative drag error'),('drag_correlation',['rho_d'],['Drag'],r'Spearman $\rho$')]
                for name,keys,labels,ylabel in groups:
                    present=[(k,l) for k,l in zip(keys,labels) if math.isfinite(number(metrics.get(k)))]
                    if present:self.bars(target/name,[number(metrics[k]) for k,l in present],[l for k,l in present],info['task_title'],ylabel,caption,correlation=name.endswith('correlation'))
                if task=='airfrans' and isinstance(metrics.get('score'),dict):
                    try:self.air_scores(target,metrics['score'],caption)
                    except (ValueError,TypeError) as e:self.warn(f'{directory}: malformed score axes ({e}); raw table retained')
            if task=='car':self.warn(f'{directory}: original Car evaluator saves no per-case drag coefficient pairs or geometry/masks; no inferred drag parity or new spatial field plot')
            if task=='airfrans':
                try:self.air_coefficients(directory,target,caption)
                except (ValueError,OSError,TypeError,IndexError) as e:self.warn(f'{directory}: coefficient plots unavailable ({e})')
            self.copy_assets(directory,target/'original_figures',recursive=False)

    def air_scores(self,target,score,caption):
        for key,stdkey,labels,ylabel in (
            ('mean_score_vol','std_score_vol',['vx','vy','p','nut'],'Volume MSE (normalized)'),
            ('mean_score_surf','std_score_surf',['vx','vy','p','nut'],'Surface MSE (normalized)'),
            ('mean_score_force','std_score_force',['Drag','Lift'],'Mean relative coefficient error'),
            ('spearman_coef_mean','spearman_coef_std',['Drag','Lift'],r'Spearman $\rho$')):
            if key not in score:continue
            values=np.asarray(score[key],dtype=float);std=np.asarray(score[stdkey],dtype=float) if stdkey in score else None
            if values.ndim!=2 or values.shape[1]!=len(labels) or (std is not None and std.shape!=values.shape):
                self.warn(f'{target}: unexpected {key} axes; table retained, no guessed reshape');continue
            for i,v in enumerate(values):
                self.bars(target/(key+f'_model_{i:02d}'),v,labels,f'Model group {i}',ylabel,
                          caption+' Error bars, when present, are the saved population standard deviations over model members, not confidence intervals or repeated test runs.',std=None if std is None else std[i],correlation=key.startswith('spearman'))

    def load_array(self,path):
        # allow_pickle=False is intentional: legacy object/ragged arrays are not executable input.
        self.track(path)
        return np.load(path,allow_pickle=False)

    def air_coefficients(self,directory,target,caption):
        truthpath=directory/'true_coefs.npy';predpath=directory/'pred_coefs_mean.npy'
        if not truthpath.exists() or not predpath.exists():return
        gt,pred=self.load_array(truthpath),self.load_array(predpath)
        if gt.ndim!=2 or gt.shape[1]!=2 or pred.ndim!=3 or pred.shape[0]!=len(gt) or pred.shape[2]!=2:raise ValueError('expected true[N,2], pred[N,model_groups,2]')
        stdpath=directory/'pred_coefs_std.npy';std=self.load_array(stdpath) if stdpath.exists() else None
        if std is not None and std.shape!=pred.shape:raise ValueError('coefficient std axes mismatch')
        numeric=[]
        for group in range(pred.shape[1]):
            for c,label in enumerate(('Drag coefficient $C_D$','Lift coefficient $C_L$')):
                x,y=gt[:,c],pred[:,group,c];valid=np.isfinite(x)&np.isfinite(y)
                numeric.extend(dict(case=i,model_group=group,channel=('drag','lift')[c],reference=float(x[i]),prediction=float(y[i]),saved_std=None if std is None else float(std[i,group,c])) for i in range(len(x)))
                if not valid.any():continue
                fig,ax=plt.subplots(figsize=(3.5,3.1),layout='constrained')
                ax.scatter(x[valid],y[valid],s=12,c='#0072B2',alpha=.7,edgecolors='none')
                if std is not None:
                    s=std[:,group,c];mask=valid&np.isfinite(s)&(s>=0)
                    ax.errorbar(x[mask],y[mask],yerr=s[mask],fmt='none',ecolor='#0072B2',alpha=.25,lw=.6)
                lo=min(x[valid].min(),y[valid].min());hi=max(x[valid].max(),y[valid].max());pad=max(hi-lo,1e-6)*.06
                ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],'--',color='#555555',lw=.8,label='Parity')
                ax.set(xlabel='Reference',ylabel='Predicted member mean',title=label,xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad));ax.set_aspect('equal');ax.legend(frameon=False)
                self.save_figure(fig,target/f'coefficient_{group}_{c}',caption+' Coefficients are saved member means; error bars are saved member std, not a confidence interval. The dashed line denotes exact agreement.')
        write_csv(target/'coefficients.csv',numeric,['case','model_group','channel','reference','prediction','saved_std'])
        # Original surface arrays: true[2,N,2], prediction[groups,2,N,2].
        for path in sorted(directory.glob('true_surf_coefs_*.npy')):
            index=path.stem.removeprefix('true_surf_coefs_');pp=directory/f'surf_coefs_{index}.npy'
            if not pp.exists():continue
            a,b=self.load_array(path),self.load_array(pp)
            if a.ndim!=3 or a.shape[0]!=2 or a.shape[-1]!=2 or b.ndim!=4 or b.shape[1:]!=a.shape:
                self.warn(f'{path}: unexpected surface axes; skipped');continue
            for group in range(len(b)):
                fig,axes=plt.subplots(1,2,figsize=(7.16,2.7),layout='constrained')
                data=[]
                for c,(ax,label) in enumerate(zip(axes,(r'$C_p$',r'$C_f$'))):
                    ax.scatter(a[c,:,0],a[c,:,1],s=7,color='#333333',label='Reference')
                    ax.scatter(b[group,c,:,0],b[group,c,:,1],s=7,color='#D55E00',marker='x',linewidths=.5,label='Prediction')
                    ax.set(xlabel=r'$x/c$',ylabel=label);ax.legend(frameon=False);ax.grid(alpha=.15)
                    if c==0:ax.invert_yaxis()
                    for kind,arr in [('reference',a[c]),('prediction',b[group,c])]:
                        data.extend(dict(channel=('Cp','Cf')[c],kind=kind,x=float(x),value=float(y)) for x,y in arr)
                self.save_figure(fig,target/f'surface_{index}_group_{group}',caption+' Saved surface coefficients; Cp axis inverted by aerodynamic convention. Native points are not joined across upper/lower surfaces.')
                write_csv(target/f'surface_{index}_group_{group}.csv',data,['channel','kind','x','value'])

    def copy_assets(self,source,target,recursive=True):
        if not source.is_dir():return
        extensions={'.pdf','.png','.svg','.txt','.tex','.json'}
        if self.args.include_arrays:extensions|={'.npz','.npy'}
        paths=source.rglob('*') if recursive else source.glob('*')
        for f in sorted(paths):
            if not f.is_file() or f.is_symlink() or f.suffix.lower() not in extensions:continue
            # Flat evaluations copy figures, not arbitrary metadata/checkpoints.
            if not recursive and f.suffix.lower() not in {'.pdf','.png','.svg'}:continue
            to=target/f.relative_to(source);to.parent.mkdir(parents=True,exist_ok=True)
            raw=self.track(f);to.write_bytes(raw)

    def fields(self,run,dest):
        if self.args.fields=='none':return
        root=run/'visualizations';found=False
        for member in sorted(root.glob('member_*')):
            epochs=sorted((p for p in member.glob('epoch_*') if p.is_dir() and p.name[6:].isdigit()),key=lambda p:int(p.name[6:]))
            if self.args.fields=='latest':epochs=epochs[-1:]
            for epoch in epochs:
                self.copy_assets(epoch,dest/'field_snapshots'/member.name/epoch.name);found=True
            if epochs:
                self.copy_assets(member/'scales',dest/'field_scales'/member.name)
                if (member/'events.jsonl').exists():
                    raw=self.track(member/'events.jsonl');d=dest/'field_snapshots'/member.name/'events.jsonl';d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(raw)
                    self.warn(f'{run}: field snapshots are stored diagnostics; consult {member.name}/events.jsonl for partial/failed exports')
        if not found:self.warn(f'{run}: no stored periodic field snapshots; geometry is not fabricated from checkpoint or coordinate-free predictions')

    def report_run(self,run,task,config,architecture):
        args=config.get('resolved_arguments',{});arch=config.get('architecture') or architecture.get('architecture',{})
        model=config.get('model') or arch.get('family') or arch.get('model_name') or 'unknown'
        seed=args.get('seed');rid=slug(run.name)+'_'+hashlib.sha256(str(run).encode()).hexdigest()[:8]
        status=self.json(run/'train_results.json',{})
        info=dict(id=rid,source=str(run),task=task,task_title=TITLES.get(task,task.title()),model=model,seed=seed,
                  training_status=status.get('status','unrecorded'),parameters=config.get('parameters'),architecture=arch,
                  fold=args.get('fold_id'),split=args.get('task'),
                  front_latent_mode=arch.get('front_latent_mode'),history_mode=arch.get('history_mode'),cdpa_mode=arch.get('cdpa_mode'))
        selection=next((r for r in self.selection if r['source']==str(run)),{})
        info.update(selection=selection,final_validation_loss=selection.get('final_validation_loss'),
                    validation_epochs=selection.get('validation_epochs'))
        mode=', '.join(f'{k}={v}' for k,v in [('front',info['front_latent_mode']),('history',info['history_mode']),('CDPA',info['cdpa_mode'])] if v is not None)
        info['label']=f'{info["task_title"]} / {model} / seed {seed if seed is not None else "unrecorded"}'+(' / '+mode if mode else '')
        if info['fold'] is not None:info['label']+=f'; fold {info["fold"]}'
        if info['split'] is not None:info['label']+=f'; split {info["split"]}'
        dimensions=', '.join(f'{k}={arch[k]}' for k in ('L','F','M','kernel_rank','d','d_model') if k in arch)
        if dimensions:info['label']+='; '+dimensions
        dest=self.directory/task/rid;dest.mkdir(parents=True)
        self.runs.append(info);write_json(dest/'run_summary.json',info)
        for name in ('config.json','architecture.json','task.json','train_results.json','eval_results.json'):
            path=run/name
            if path.is_file():(dest/name).write_bytes(self.track(path))
        self.training(run,dest,task,info,config);self.test_results(run,dest,task,info);self.fields(run,dest)
        # Preserve legacy generated loss figures and final log records, with no invented epoch history.
        self.copy_assets(run,dest/'legacy_figures',recursive=False)
        for member in sorted(run.glob('member_*')):
            if member.is_dir():
                self.copy_assets(member,dest/'legacy_figures'/member.name,recursive=False)
                for f in member.glob('*_log.json'):
                    d=dest/'legacy_records'/member.name/f.name;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(self.track(f))
        for pattern in ('*_log.json','log_*.json'):
            for f in sorted(run.glob(pattern)):
                d=dest/'legacy_records'/f.name;d.parent.mkdir(exist_ok=True);d.write_bytes(self.track(f))
        self.make_index(dest,info['label'])

    def make_index(self,directory,title):
        images=sorted((p for p in directory.rglob('*.png') if p.is_file()),
                      key=lambda p:(not p.name.startswith('pde_selected_'),str(p)))
        cards=[]
        for p in images:
            rel=p.relative_to(directory).as_posix();pdf=p.with_suffix('.pdf');caption=p.with_suffix('.txt')
            if not caption.exists() and p.name in ('fields.png','curve.png'):
                caption=p.parent/'caption.txt'
            content=caption.read_text() if caption.exists() else 'Existing saved figure; see accompanying metadata/caption for its scope.'
            cards.append('<section><h3>'+html.escape(rel)+'</h3><a href="'+html.escape(rel,quote=True)+'"><img loading="lazy" src="'+html.escape(rel,quote=True)+'"></a><p>'+html.escape(content)+'</p>'+('<a href="'+html.escape(pdf.relative_to(directory).as_posix(),quote=True)+'">PDF</a>' if pdf.exists() else '')+'</section>')
        links=[]
        for p in sorted(directory.rglob('*')):
            if p.is_file() and p.suffix in ('.csv','.json','.tex'):
                rel=p.relative_to(directory).as_posix();links.append('<li><a href="'+html.escape(rel,quote=True)+'">'+html.escape(rel)+'</a></li>')
        page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.65 Georgia,serif;color:#243544;background:#f8fafb}h1{font-size:28px}section{background:white;border:1px solid #dde3e9;border-radius:8px;padding:20px;margin:24px 0}img{max-width:100%;height:auto}h3{overflow-wrap:anywhere;font:14px monospace}a{color:#006598}p{max-width:90ch}</style><h1>'+html.escape(title)+'</h1><p>One selected run per dataset, using its final recorded validation loss. See selection.json/CSV for candidates, seeds, epochs, losses and exclusions. Curves are unsmoothed; selected-run members and evaluation attempts remain separate. No test values are inferred from validation and no source files are changed.</p>'+''.join(cards)+'<h2>Numeric tables and provenance</h2><ul>'+''.join(links)+'</ul></html>'
        (directory/'index.html').write_text(page,encoding='utf-8')


def infer_task(run,config,architecture,allowed,explicit_task=None):
    arch=architecture.get('architecture',{})
    task=config.get('task') or architecture.get('task') or arch.get('task_name')
    if task:task=ALIASES.get(str(task).lower(),str(task).lower())
    if not task:
        known={task for tasks in PROJECTS.values() for task in tasks}
        matches={ALIASES.get(part.lower(),part.lower()) for part in run.parts}&known
        if len(matches)==1:task=matches.pop()
    if not task:
        for project,tasks in PROJECTS.items():
            if project in run.parts and len(tasks)==1:task=tasks[0]
    if not task:task=explicit_task
    return task


def discover(roots):
    result=set()
    for root in roots:
        if not root.is_dir():continue
        for current,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in ('visualizations','evaluations','training_artifacts','result_visualizations','__pycache__','.git','_seed_suites') and not Path(current,d).is_symlink())
            if any(f in files for f in ('config.json','train_history.jsonl','train_results.json','architecture.json','score.json','train.log')) or any(f.endswith('_log.json') or re.fullmatch(r'log_\d+\.json',f) for f in files):
                result.add(Path(current).resolve());dirs[:]=[]
    return sorted(result)


def parser():
    p=argparse.ArgumentParser(description='Offline report + ZIP for the lowest-final-validation run per dataset; never loads datasets, models or checkpoints.')
    p.add_argument('project',choices=PROJECTS)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2],help='Actual checkout, defaults to script location (not /home/hwz)')
    p.add_argument('--runs-root',type=Path,action='append',help='Scan this root; repeatable. Overrides default output/runs/project metrics+scores roots')
    p.add_argument('--run-dir',type=Path,action='append',help='Exact run directory; repeatable; disables automatic discovery')
    p.add_argument('--task',choices=PROJECTS['PDE-Solving-StandardBenchmark'],help='Restrict PDE tasks; also identifies an explicit legacy run lacking task metadata')
    p.add_argument('--model',help='Filter saved model family, e.g. kcdno/CDLNO/lrsa_matched')
    p.add_argument('--seed',type=int,help='Filter the explicitly recorded seed; never infer it from a filename')
    p.add_argument('--output-dir',type=Path,help='New bundle directory; existing paths are rejected')
    p.add_argument('--fields',choices=['latest','all','none'],default='latest',help='Copy latest stored epoch per member of the selected run; all copies all stored epochs of that run')
    p.add_argument('--include-arrays',action='store_true',help='Include copied field NPZ/NPY arrays (can be large)')
    p.add_argument('--dpi',type=int,default=600,choices=(300,600))
    p.add_argument('--dry-run',action='store_true',help='Read/discover only; no plots or output directories')
    return p


def main(argv=None):
    args=parser().parse_args(argv);repo=args.repo_root.resolve();project=repo/args.project
    if not project.is_dir():raise ValueError(f'project directory does not exist: {project}')
    if args.task and args.task not in PROJECTS[args.project]:raise ValueError('--task is for the six PDE benchmarks only')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_'+uuid.uuid4().hex[:8]
    output=(args.output_dir or project/'result_visualizations'/stamp).resolve()
    archive=output.parent/(output.name+'.zip')
    if output.exists() or archive.exists():raise FileExistsError(f'output already exists: {output}')
    roots=args.runs_root or [Path(os.environ.get('CDLNO_RUNS_ROOT',repo/'output')),repo/'runs',project/'metrics',project/'scores']
    if args.run_dir:
        candidates=sorted({p.resolve() for p in args.run_dir})
        for p in candidates:
            if not p.is_dir():raise ValueError(f'run directory does not exist: {p}')
    else:candidates=discover([p.resolve() for p in roots])
    bundle=Bundle(args,output);selected=[]
    for run in candidates:
        cfg=bundle.json(run/'config.json',{});arch=bundle.json(run/'architecture.json',{})
        if not isinstance(cfg,dict) or not isinstance(arch,dict):bundle.warn(f'{run}: invalid config/architecture object');continue
        task=infer_task(run,cfg,arch,PROJECTS[args.project],(args.task or (PROJECTS[args.project][0] if len(PROJECTS[args.project])==1 else None)) if args.run_dir else None)
        if task not in PROJECTS[args.project] or (args.task and task!=args.task):continue
        resolved=cfg.get('architecture') or arch.get('architecture',{})
        model=cfg.get('model') or resolved.get('family') or resolved.get('model_name') or 'unknown'
        if args.model and model!=args.model:continue
        if args.seed is not None and cfg.get('resolved_arguments',{}).get('seed')!=args.seed:continue
        if output==run or output.is_relative_to(run) or run.is_relative_to(output):raise ValueError('output must not overlap an input run directory')
        selected.append((run,task,cfg,arch))
    candidate_count=len(selected);selected=bundle.select_runs(selected)
    print('Project:',project);print('Report:',output);print('Candidates:',candidate_count);print('Runs:',len(selected))
    print('Selection:',SELECTION_POLICY)
    for record in bundle.selection:
        state='SELECTED' if record['selected'] else 'EXCLUDED'
        print(f'  {state} {record["task"]} seed={record["seed"]} final_validation={record["final_validation_loss"]} '
              f'epochs={record["validation_epochs"]}: {record["source"]} ({record["reason"]})')
    if args.dry_run:return 0
    output.mkdir(parents=True,exist_ok=False)
    with plt.rc_context(STYLE):
        for run,task,cfg,arch in selected:
            bundle.report_run(run,task,cfg,arch)
        if args.project=='PDE-Solving-StandardBenchmark':
            bundle.pde_overview();bundle.pde_evaluation_overview()
    if not selected:bundle.warn('No eligible saved runs with final validation. No synthetic curves or metrics were created. See selection.json and source history/status; --run-dir/--runs-root locate custom runs.')
    write_json(output/'selection.json',dict(policy=SELECTION_POLICY,candidates=bundle.selection))
    write_csv(output/'selection.csv',bundle.selection,['task','model','seed','fold','split','training_status','selected','eligible','metric','final_validation_loss','validation_epochs','member_losses','weights','last_training_epochs','reason','source'])
    write_csv(output/'runs.csv',bundle.runs,['id','task','model','seed','fold','split','training_status','parameters','front_latent_mode','history_mode','cdpa_mode','final_validation_loss','validation_epochs','source'])
    write_csv(output/'training_history.csv',bundle.epoch_rows,['run','task','model','seed','member','epoch','metric','value'])
    write_csv(output/'evaluation_metrics.csv',bundle.eval_rows,['run','task','model','seed','evaluation','status','metric','value'])
    manifest=dict(format_version=2,generated_at_utc=stamp,project=args.project,repo_root=str(repo),runs=bundle.runs,selection=bundle.selection,selection_policy=SELECTION_POLICY,
                  warnings=bundle.warnings,sources=bundle.sources,options={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v for k,v in vars(args).items()},
                  environment=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__),style= {k:v for k,v in STYLE.items() if k!='axes.prop_cycle'},
                  semantics='Read-only records; no smoothing, no averaging across runs/seeds/folds, no dataset/checkpoint load; status is taken from saved files and can be stale after hard termination.')
    write_json(output/'manifest.json',manifest)
    (output/'WARNINGS.txt').write_text('\n'.join(bundle.warnings) + '\n',encoding='utf-8')
    (output/'README.md').write_text('# '+args.project+' results\n\nOpen index.html for the gallery. PDF/PNG figures have English TXT/LaTeX captions. CSV files preserve metric names and epoch/member/seed/evaluation axes. Sources and SHA256 hashes are in manifest.json.\n\n'+SELECTION_POLICY+'\n\nselection.json/CSV record all candidates; curves, field images and evaluation tables include selected runs only. PDE pde_selected_training/evaluation.pdf each show all six tasks in one figure.\n\nReports contain saved records, not fresh evaluation. Missing histories cannot be recovered from checkpoints. No model weights are included. See WARNINGS.txt before interpreting figures.\n',encoding='utf-8')
    bundle.make_index(output,args.project+' / experiment report')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,arcname=output.name+'/'+p.relative_to(output).as_posix())
    print('Completed:',output/'index.html');print('ZIP:',archive);print('Notes:',len(bundle.warnings))
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (ValueError,OSError) as error:print('Report error:',error,file=sys.stderr);sys.exit(2)
