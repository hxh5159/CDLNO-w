"""Generate explicitly synthetic field-rendering examples; no model/data IO."""
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from cdlno.visualization import fixed_scales, render_fields


out = Path(__file__).resolve().parent
artifact = Path(tempfile.mkdtemp(prefix='viz-v1-demo-', dir='/home/hwz/CDLNO-artifacts'))
started = time.perf_counter()
y, x = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 2, 7), indexing='ij')
xy = np.column_stack(((x + .1*np.sin(2*y)).ravel(), y.ravel()))
truth = (np.sin(np.pi*x)*np.cos(np.pi*y)).reshape(-1, 1)
prediction = truth + .1*np.cos(2*x).reshape(-1, 1)
scales = fixed_scales(artifact/'scales.json', truth, prediction, ['u'])
render_fields(artifact/'grid_5x7', coordinates=xy, truth=truth, prediction=prediction,
              channel_names=['u'], scales=scales, task='SYNTHETIC ONLY',
              case_id='analytical field + prescribed perturbation', completed_epoch=50, grid_shape=(5,7),
              metadata={'not_model_predictions': True, 'not_dataset_samples': True})
phi, theta = np.meshgrid(np.linspace(.1, np.pi-.1, 12), np.linspace(0, 2*np.pi, 24, endpoint=False), indexing='ij')
xyz = np.column_stack((np.sin(phi).ravel()*np.cos(theta).ravel(),
                       np.sin(phi).ravel()*np.sin(theta).ravel(), np.cos(phi).ravel()))
truth = xyz[:, :1]
prediction = truth + .15*xyz[:, 1:2]**2
render_fields(artifact/'cloud_3d', coordinates=xyz, truth=truth, prediction=prediction,
              channel_names=['scalar'], scales=fixed_scales(artifact/'scales3d.json', truth, prediction, ['scalar']),
              task='SYNTHETIC ONLY', case_id='analytical field + prescribed perturbation', completed_epoch=100,
              metadata={'not_model_predictions': True, 'not_dataset_samples': True})
result = dict(artifact=str(artifact), elapsed_seconds=time.perf_counter()-started,
              files={str(p.relative_to(artifact)):dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                     for p in sorted(artifact.rglob('*')) if p.is_file()})
(out/'render-demo.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
