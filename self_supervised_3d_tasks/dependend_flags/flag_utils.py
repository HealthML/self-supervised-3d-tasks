import inspect
from functools import partial as func_partial
from ..errors import MappingRedefinitionError


def check_for_missing_arguments(model_fn, kwargs):
    model_fn_parameters = inspect.signature(model_fn).parameters.values()
    missing_flags = []
    for param in model_fn_parameters:
        if param.default is param.empty:
            # the argument has no default value --> required parameter --> check if given!
            if not param.name in kwargs:
                missing_flags.append(param.name)
    return missing_flags


def collect_model_kwargs(model_fn, kwargs, allow_remapping=False):
    """I collect all applicable keyword arguments.

    Args:
        model_fn: the function for which we collect the arguments
        kwargs: the "pool" of kwargs that could be mapped
        allow_remapping: If the model_fn is a functools.partial, should I allow the redefinition of a model_fn argument

    Returns:

    """
    model_fn_parameters = inspect.signature(model_fn).parameters.values()
    already_mapped_params = []
    if isinstance(model_fn, func_partial):
        already_mapped_params += (
            model_fn.keywords.keys()
        )  # collect all already mapped keywords
        # collect all already mapped positionals
        positionals = [
            param.name for param in model_fn_parameters if param.default is param.empty
        ]
        already_mapped_params += positionals[: len(model_fn.args)]

    kwargs_collected = {}
    for param in model_fn_parameters:
        if not allow_remapping:
            if param.name in already_mapped_params:
                raise MappingRedefinitionError(param.name)
        if param.name in kwargs:
            kwargs_collected[param.name] = kwargs[param.name]
    return kwargs_collected
