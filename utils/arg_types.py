import json

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


def module_enum(*mods):
    mod_dict = {k.lower(): v for mod in reversed(mods) for k, v in vars(mod).items()}
    f = lambda s: mod_dict[s.lower()]
    joiner = f".\\*/"
    f.__name__ = f"{joiner.join([mod.__name__ for mod in mods])}.\\*"
    return f


json_type = json.loads
json_type.__name__ = "json"
