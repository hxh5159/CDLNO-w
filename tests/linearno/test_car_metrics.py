"""L7 known evaluation contracts; does not resolve paper training objective."""
import copy
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

from cdlno.linearno.car_metrics import (coefficient, drag_backend, drag_pair,
                                      field_metrics, fields, relative_l2)


class CarMetricChecks(unittest.TestCase):
    def inputs(self):
        prediction=torch.arange(28,dtype=torch.float64).reshape(7,4)/10
        truth=prediction+torch.tensor([.2,.4,.8,1.6])
        surf=torch.tensor([False,True,False,False,True,False,True])
        coef=[np.zeros(7),np.ones(7),np.array([4.,5.,6.,7.]),np.array([2.,3.,4.,5.])]
        return prediction,truth,surf,coef

    def test_independent_field_metrics_mask_channels_decode_and_no_grad(self):
        out,y,surf,coef=self.inputs();out.requires_grad_()
        actual=field_metrics(out,y,surf,coef)
        a=out.detach().numpy()*coef[3]+coef[2];b=y.numpy()*coef[3]+coef[2];mask=surf.numpy()
        expected=dict(physical_relative_l2_pressure=np.linalg.norm(a[mask,3]-b[mask,3])/np.linalg.norm(b[mask,3]),
            physical_relative_l2_velocity=np.linalg.norm(a[~mask,:3]-b[~mask,:3])/np.linalg.norm(b[~mask,:3]),
            normalized_mse_pressure=((out.detach().numpy()[mask,3]-y.numpy()[mask,3])**2).mean(),
            normalized_mse_velocity_components=((out.detach().numpy()[~mask,:3]-y.numpy()[~mask,:3])**2).mean(0))
        for key,value in actual.items():
            np.testing.assert_allclose(value.numpy(),expected[key],atol=1e-12,rtol=1e-12)
            self.assertFalse(value.requires_grad)
        changed=out.detach().clone();changed[~surf,3]+=1000
        other=field_metrics(changed,y,surf,coef)
        for key in actual:torch.testing.assert_close(actual[key],other[key],atol=0,rtol=0)

    def test_denominator_mask_and_normalizer_validation(self):
        out,y,surf,coef=self.inputs()
        with self.assertRaisesRegex(ValueError,'nonzero'): relative_l2(torch.ones(2),torch.zeros(2))
        with self.assertRaisesRegex(ValueError,'nonempty'): relative_l2(torch.ones(0),torch.ones(0))
        for mask in (torch.zeros_like(surf),torch.ones_like(surf)):
            with self.assertRaisesRegex(ValueError,'both'):field_metrics(out,y,mask,coef)
        with self.assertRaisesRegex(ValueError,'boolean'):field_metrics(out,y,surf.int(),coef)
        with self.assertRaisesRegex(ValueError,'saved coef_norm'):field_metrics(out,y,surf,None)
        invalid=copy.deepcopy(coef);invalid[3][0]=-1
        with self.assertRaisesRegex(ValueError,'std>=0'):field_metrics(out,y,surf,invalid)

    def test_drag_receives_surface_velocity_and_actual_nonzero_fold_path(self):
        out,y,surf,coef=self.inputs();data=Data(x=torch.zeros(7,7),y=y,surf=surf)
        # A pure routing assertion: no fake dataset directory or VTK file created.
        sample=Path('/synthetic-path-only/param4/explicit_sample');calls=[]
        def capture(path,pressure,velocity):
            calls.append((path,pressure.copy(),velocity.copy()));return float(len(calls))
        result=drag_pair(data,out,coef,sample,coefficient_fn=capture)
        self.assertEqual(result,(1.,2.));self.assertEqual(len(calls),2)
        physical,truth=fields(out,y,surf,coef)
        for call,target in zip(calls,(physical,truth)):
            self.assertEqual(call[0],sample)
            np.testing.assert_array_equal(call[1],target[surf,3:4].numpy())
            np.testing.assert_array_equal(call[2],target[surf,:3].numpy())
            self.assertEqual(len(call[1]),3)  # volume has4 points; cannot substitute

    def test_real_VTK_primitives_match_old_drag_arithmetic_and_validate_count(self):
        backend=drag_backend();vtk=backend.vtk
        def mesh():
            points=vtk.vtkPoints()
            for p in ((0,0,0),(1,0,0),(1,1,0),(0,1,0)):points.InsertNextPoint(*p)
            cell=vtk.vtkQuad()
            for i in range(4):cell.GetPointIds().SetId(i,i)
            result=vtk.vtkUnstructuredGrid();result.SetPoints(points);result.InsertNextCell(cell.GetCellType(),cell.GetPointIds())
            return result
        prediction=np.array([[1.],[2.],[3.],[4.]]);velocity=np.ones((4,3));paths=[]
        def load(path):paths.append(path);return mesh()
        sample=Path('/synthetic-routing-only/param4/case')
        # Real in-memory VTK math, isolated file-presence/read seam; no real data.
        with patch.object(backend,'load_unstructured_grid_data',load),patch.object(Path,'is_file',return_value=True):
            actual=coefficient(sample,prediction,velocity,_backend=backend)
            self.assertEqual(paths,[str(sample/'quadpress_smpl.vtk')])
            expected=backend.cal_coefficient('case',prediction,velocity)
            self.assertEqual(actual,expected)
            self.assertTrue(np.isfinite(actual))
            with self.assertRaisesRegex(ValueError,'count'):
                coefficient(sample,prediction[:3],velocity[:3],_backend=backend)
            with self.assertRaisesRegex(ValueError,'matched surface'):
                coefficient(sample,prediction,velocity[:3],_backend=backend)
        with patch.object(Path,'is_file',return_value=False),self.assertRaisesRegex(FileNotFoundError,'actual sample path'):
            coefficient(sample,prediction,velocity,_backend=backend)
