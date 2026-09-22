"""Thin V3 view over the repository's existing experiment recorder."""
from linearno_loop.v3.config import validate_config


class V3OutputAdapter:
    """Delegate observation calls; owns no trainer, metric, plot, or directory."""
    def __init__(self, recorder, config):
        self.recorder = recorder
        self.config = validate_config(config)
        required = ("attach_model", "record_training_setup", "record_epoch",
                    "visualize", "record_metrics")
        missing = [name for name in required if not callable(getattr(recorder, name, None))]
        if missing:
            raise TypeError("recorder lacks current output interface: " + ", ".join(missing))

    @property
    def result_dir(self):
        return self.recorder.result_dir

    def attach_model(self, model, **kwargs):
        if getattr(model, "config_hash", None) != self.config["config_hash"]:
            raise ValueError("output model differs from V3 resolved config")
        return self.recorder.attach_model(model, **kwargs)

    def record_training_setup(self, *args, **kwargs):
        return self.recorder.record_training_setup(*args, **kwargs)

    def record_epoch(self, *args, **kwargs):
        return self.recorder.record_epoch(*args, **kwargs)

    def visualize(self, *args, **kwargs):
        return self.recorder.visualize(*args, **kwargs)

    def record_metrics(self, *args, **kwargs):
        return self.recorder.record_metrics(*args, **kwargs)

