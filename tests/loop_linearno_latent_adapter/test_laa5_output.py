import tempfile
import unittest
from pathlib import Path

from .wrapper_support import config, construct


class Recorder:
    def __init__(self, directory): self.directory, self.calls = Path(directory), []
    def attach_model(self, *args, **kwargs): self.calls.append(("attach", args, kwargs)); return "attached"
    def record_training_setup(self, *args, **kwargs): self.calls.append(("setup", args, kwargs))
    def record_epoch(self, *args, **kwargs): self.calls.append(("epoch", args, kwargs))
    def visualize(self, *args, **kwargs): self.calls.append(("visualize", args, kwargs)); return "visualized"
    def record_metrics(self, *args, **kwargs): self.calls.append(("metrics", args, kwargs))
    @property
    def result_dir(self): return str(self.directory / "results") + "/"


class OutputTests(unittest.TestCase):
    def test_delegates_current_recorder_without_parallel_math(self):
        from cdlno.linearno_loop.v3.output import V3OutputAdapter
        with tempfile.TemporaryDirectory() as temp:
            c = config("elasticity"); recorder = Recorder(temp)
            adapter = V3OutputAdapter(recorder, c)
            model = construct(c)
            self.assertEqual(adapter.attach_model(model, member=2), "attached")
            adapter.record_training_setup("optimizer", "scheduler")
            adapter.record_epoch(1, {"loss": 1.0}, member=2)
            self.assertEqual(adapter.visualize(model, 1, 2, member=2, dataset="fixture"), "visualized")
            adapter.record_metrics({"relative_l2": 1.0})
            self.assertEqual(adapter.result_dir, recorder.result_dir)
            self.assertEqual([row[0] for row in recorder.calls],
                             ["attach", "setup", "epoch", "visualize", "metrics"])

    def test_exceptions_propagate_and_run_directory_is_never_reserved(self):
        from cdlno.linearno_loop.v3.output import V3OutputAdapter
        with tempfile.TemporaryDirectory() as temp:
            recorder = Recorder(temp); adapter = V3OutputAdapter(recorder, config("darcy"))
            def fail(*args, **kwargs): raise RuntimeError("recorder failure")
            recorder.visualize = fail
            with self.assertRaisesRegex(RuntimeError, "recorder failure"):
                adapter.visualize(construct(config("darcy")), 1, 2)
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == "__main__": unittest.main()
