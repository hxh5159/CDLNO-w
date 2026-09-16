"""Deterministic synthetic layout previews, NEVER model/dataset results.
Run from the repository root: python -m docs.periodic_visualization_audit.make_demos --output NEW_DIRECTORY
"""
import argparse
from pathlib import Path
import numpy as np
from cdlno.visualization import ColorScale,render_fields,render_curves


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);out=Path(p.parse_args().output)
    def render(name,coords,truth,pred,**kw):
        return render_fields(out/name,coordinates=coords,truth=truth,prediction=pred,
            channel_names=['Scalar field $u$'],scales=[ColorScale(float(truth.min()),float(truth.max()),.2*float(np.ptp(truth)))],
            task='synthetic layout demonstration',case_id='0',completed_epoch=50,
            model_name='Synthetic prediction',metadata={'seed':0,'caption_note':'SYNTHETIC ILLUSTRATION ONLY. Analytic fields with prescribed perturbations; not trained-model predictions or dataset measurements.'},**kw)
    x,y=np.meshgrid(np.linspace(0,1,61),np.linspace(0,1,81),indexing='ij')
    coords=np.c_[x.ravel(),(y+.06*np.sin(2*np.pi*x)*np.sin(np.pi*y)).ravel()]
    truth=(np.sin(np.pi*x)*np.sin(2*np.pi*y)).reshape(-1,1)
    prediction=truth+.07*np.cos(3*np.pi*coords[:,:1])*np.sin(2*np.pi*coords[:,1:])
    render('curved_grid',coords,truth,prediction,grid_shape=x.shape)
    rng=np.random.default_rng(0);xyz=rng.uniform(-1,1,(5000,2));xyz=xyz[(xyz**2).sum(1)>.2**2]
    truth=(1+np.exp(-4*(xyz[:,0]**2+xyz[:,1]**2))+.4*xyz[:,0])[:,None]
    render('point_cloud_hole',xyz,truth,truth+.06*np.cos(8*xyz[:,:1]))
    a,b=np.meshgrid(np.linspace(0,2*np.pi,100),np.linspace(.03,np.pi-.03,50),indexing='ij')
    xyz=np.c_[2*np.sin(b.ravel())*np.cos(a.ravel()),np.sin(b.ravel())*np.sin(a.ravel()),.6*np.cos(b.ravel())]
    truth=(xyz[:,0]+.3*xyz[:,2])[:,None]
    render('surface_3d',xyz,truth,truth+.12*np.sin(3*xyz[:,1:2]))
    render_curves(out/'time_curve',np.arange(1,11),{'Synthetic error':np.linspace(.025,.11,10)},xlabel='Forecast step (synthetic)',ylabel='Relative $L_2$ error')


if __name__=='__main__':main()
