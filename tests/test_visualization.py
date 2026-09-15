"""V1 pure field rendering, point order, color/metric and side-effect checks."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from cdlno.visualization import ColorScale,fixed_scales,render_fields


class VisualizationChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)
        row,col=np.meshgrid(np.arange(5),np.arange(7),indexing='ij')
        self.xyz=np.stack((col.ravel(),row.ravel()),-1)
        self.gt=(10*row+col).reshape(-1,1).astype(float)
        self.pred=self.gt+np.arange(35).reshape(-1,1)/10

    def test_non_square_grid_png_pdf_npz_and_locked_scales(self):
        scale_path=self.path/'scale.json'
        scales=fixed_scales(scale_path,self.gt,self.pred,['u'])
        before=scale_path.read_bytes()
        again=fixed_scales(scale_path,self.gt,self.pred*100,['u'])
        self.assertEqual(scales,again);self.assertEqual(scale_path.read_bytes(),before)
        case=self.path/'case'
        result=render_fields(case,coordinates=self.xyz,truth=self.gt,prediction=self.pred,
            channel_names=['u'],scales=scales,task='synthetic',case_id='0',completed_epoch=50,grid_shape=(5,7))
        with np.load(case/'fields.npz',allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['truth'].reshape(5,7),self.gt.reshape(5,7))
            np.testing.assert_array_equal(saved['coordinates'],self.xyz)
            np.testing.assert_array_equal(saved['prediction'],self.pred)
            np.testing.assert_allclose(saved['absolute_error'],np.abs(self.pred-self.gt))
        with Image.open(case/'fields.png') as image:
            self.assertGreater(image.width,1000);self.assertGreater(image.height,500)
        self.assertTrue((case/'fields.pdf').read_bytes().startswith(b'%PDF'))
        metric=result['channel_metrics'][0]
        self.assertAlmostEqual(metric['relative_l2'],np.linalg.norm(self.pred-self.gt)/np.linalg.norm(self.gt))
        self.assertEqual(metric['color_scale'],scales[0].to_dict())
        self.assertEqual(json.loads((case/'metadata.json').read_text())['grid_shape'],[5,7])
        with self.assertRaises(FileExistsError):render_fields(case,coordinates=self.xyz,truth=self.gt,prediction=self.pred,
            channel_names=['u'],scales=scales,task='synthetic',case_id='0',completed_epoch=50,grid_shape=(5,7))

    def test_unstructured_three_dimensional_mask_only_affects_rendering(self):
        xyz=np.column_stack((self.xyz,np.sin(self.xyz[:,0])))
        mask=np.arange(35)%2==0
        result=render_fields(self.path/'cloud',coordinates=xyz,truth=self.gt,prediction=self.pred,
            channel_names=['pressure'],scales=[ColorScale(0,46,1)],task='synthetic-car',case_id='0',completed_epoch=100,display_mask=mask)
        self.assertEqual(result['displayed_points'],18)
        with np.load(self.path/'cloud/fields.npz') as values:
            self.assertEqual(values['truth'].shape,(35,1));np.testing.assert_array_equal(values['display_mask'],mask)
        self.assertGreater(result['channel_metrics'][0]['error_above_color_fraction'],0)

    def test_zero_truth_nonfinite_channels_grid_and_physical_values(self):
        result=render_fields(self.path/'zero',coordinates=self.xyz,truth=self.gt*0,prediction=self.gt*0+1,
            channel_names=['u'],scales=[ColorScale(0,1,.1)],task='synthetic',case_id='0',completed_epoch=1)
        metric=result['channel_metrics'][0]
        self.assertIsNone(metric['relative_l2']);self.assertEqual(metric['relative_l2_status'],'undefined_zero_target')
        self.assertEqual(metric['error_above_color_fraction'],1.)
        for invalid in ((7,7),(35,), (True,35)):
            with self.assertRaises(ValueError):render_fields(self.path/'invalid',coordinates=self.xyz,truth=self.gt,prediction=self.pred,
                channel_names=['u'],scales=[ColorScale(0,46,3.4)],task='synthetic',case_id='0',completed_epoch=1,grid_shape=invalid)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            ColorScale.from_first_case(self.gt,self.gt*np.nan)
        with self.assertRaises(ValueError):fixed_scales(self.path/'bad.json',self.gt,self.pred,['a','b'])


if __name__=='__main__':unittest.main()
