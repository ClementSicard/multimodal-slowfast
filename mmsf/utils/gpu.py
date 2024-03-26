from typing import List, Any


def to_gpu(*args) -> None:
    for arg in args:
        # Transfer the data to the current GPU device.
        if isinstance(arg, (list,)):
            for i in range(len(arg)):
                arg[i] = arg[i].cuda(non_blocking=True)
        elif isinstance(labels, (dict,)):
            labels = {k: v.cuda() for k, v in labels.items()}
        else:
            arg = arg.cuda(non_blocking=True)
