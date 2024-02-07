import inspect


def default(val, d, null_val=None):
    return val if val != null_val else d


def get_all_subclasses_from_modules(*ms, super_cls=None, lower_case_keys=True):
    return {
        (k.lower() if lower_case_keys else k): v
        for m in ms
        for k, v in inspect.getmembers(
            m,
            lambda obj: inspect.isclass(obj)
            and (super_cls is None or issubclass(obj, super_cls)),
        )
    }
