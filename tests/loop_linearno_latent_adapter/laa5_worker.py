"""Fresh-process V3 pair reader. Metadata/config are inspected before model import."""
import json
from pathlib import Path
import sys


def main():
    directory, selector = Path(sys.argv[1]), sys.argv[2]
    from cdlno.linearno_loop.v3.checkpoint import (load_model, read_pair,
                                                    restore_training_state)
    from cdlno.linearno_loop.v3.construction import build_from_config
    import torch
    from loop_linearno_latent_adapter.wrapper_support import example, invoke
    model, metadata = load_model(directory, selector, explicit=json.loads(sys.argv[3]))
    config = metadata["resolved_config"]
    eval_output = invoke(model.eval(), config, example(config, batch=1))
    resumed = build_from_config(config)
    optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    generators = {"train": torch.Generator().manual_seed(999)}
    _, weights = read_pair(directory / "checkpoints" / (selector + ".json"),
                           expected=config)
    restore_training_state(resumed, optimizer, scheduler, metadata,
        weights=weights, generators=generators, sampler={"scope": "synthetic"},
        normalizer_spec=metadata["normalizer_spec"], scaler=None)
    resumed.train(); optimizer.zero_grad()
    resume_output = invoke(resumed, config, example(config, batch=1))
    resume_output.square().mean().backward(); optimizer.step(); scheduler.step()
    print(json.dumps(dict(class_path=type(model).__module__ + "." + type(model).__qualname__,
                               config_hash=metadata["config_hash"],
                               state_keys=list(model.state_dict()),
                               parameter_count=sum(p.numel() for p in model.parameters()),
                               eval_finite=bool(torch.isfinite(eval_output).all()),
                               resume_finite=bool(torch.isfinite(resume_output).all()),
                               restored_epoch=metadata["resume_state"]["epoch"],
                               continued_scheduler_epoch=scheduler.last_epoch)))


if __name__ == "__main__": main()
