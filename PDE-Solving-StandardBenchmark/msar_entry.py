"""Standard MSAR task integration; no data imports or experiment execution."""
from cdlno.msar_lno.standard_entry import model_kwargs, StandardRun
from cdlno.msar_lno.objective import training_forward, training_objective, ObjectiveMetrics
from cdlno.msar_lno.objective import rollout_objective
