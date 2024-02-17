import json
from utils import get_all_subclasses_from_modules

str2bool = lambda s: s.lower() in [
    "true",
    "1",
    "t",
    "y",
    "yes",
    "yeah",
    "yup",
    "certainly",
    "uh-huh",
]
str2bool.__name__ = "bool"


def dict_enum(**kwargs):
    f = lambda s: dict(**kwargs)[s]
    f.__name__ = kwargs["name"]
    return f


def tup(t):
    f = lambda s: [t(c) for c in s.split(",")]
    f.__name__ = f"{t.__name__}_tuple"
    return f


def module_enum(*mods, super_cls_filter=None):
    mod_dict = get_all_subclasses_from_modules(
        *mods, super_cls=super_cls_filter, lower_case_keys=True
    )
    f = lambda s: mod_dict.get(s.lower(), s)
    joiner = f".\\*/"
    f.__name__ = f"{joiner.join([mod.__name__ for mod in mods])}.\\*"
    return f


json_type = json.loads
json_type.__name__ = "json"
