import inspect


def check_for_missing_arguments(model_fn, kwargs):
    model_fn_parameters = inspect.signature(model_fn).parameters.values()
    missing_flags = []
    for param in model_fn_parameters:
        if isinstance(param.default, inspect._empty.__class__):
            # the argument has no default value --> required parameter --> check if given!
            if not param.name in kwargs:
                missing_flags.append(param.name)
    return missing_flags


def collect_model_kwargs(model_fn, kwargs):
    model_fn_parameters = inspect.signature(model_fn).parameters.values()
    kwargs_collected = {}
    for param in model_fn_parameters:
        if param.name in kwargs:
            kwargs_collected[param.name] = kwargs[param.name]
    return kwargs_collected
