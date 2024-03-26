from typing import Any


def to_gpu(arg) -> Any:
    # Transfer the data to the current GPU device.
    if isinstance(arg, (list,)):
        for i in range(len(arg)):
            arg[i] = arg[i].cuda(non_blocking=True)
    elif isinstance(arg, (dict,)):
        arg = {k: v.cuda() for k, v in arg.items()}
    else:
        arg = arg.cuda(non_blocking=True)

    return arg
