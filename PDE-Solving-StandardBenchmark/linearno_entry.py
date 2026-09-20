"""Six Standard-task LinearNO integration. No exp imports or data side effects."""
from cdlno.linearno.standard_entry import (
    model_kwargs, start, finish, normalizer, verify_data, StandardRun,
)

# Only the explicitly selected research family uses the new native Run subclass.
_PureStandardRun = StandardRun

def StandardRun(args, model):
    if getattr(args, 'linearno_family', None) == 'linearno_loop':
        from cdlno.linearno_loop.standard_entry import LoopStandardRun
        return LoopStandardRun(args, model)
    from cdlno.linearno_history.standard_entry import select_run
    return select_run(args, model, _PureStandardRun)
