from typing import Callable, TextIO, Dict

import argparse
import json
import yaml

supported_exts_loaders = {
    "json": json.load,
    "yml": yaml.safe_load,
    "yaml": yaml.safe_load,
}

supported_exts = list(supported_exts_loaders.keys())


def add_custom_loader(ext: str, loader: Callable[[TextIO], Dict]) -> None:
    """Add a custom loader for a specific type of config file

    Args:
        ext (str): The extension type.
        loader (Callable[[TextIO], Dict]): The loader for the extension type.
    """
    global supported_exts_loaders
    supported_exts_loaders[ext] = loader
    supported_exts.append(ext)


def __clear_defaults(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Clears the default values from a parser and sets them to argparse.SUPPRESS.

    Args:
        parser (argparse.ArgumentParser): The parser to clear defaults of.

    Returns:
        argparse.ArgumentParser: The parser with cleared defaults.
    """
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return parser


def add_config_file_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Added a config file argument to the argument parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add the config argument to.

    Returns:
        argparse.ArgumentParser: The parser with the config argument.
    """
    parser.add_argument(
        "-c",
        "--config-file",
        default=[],
        type=str,
        nargs="+",
        help=f"A config file specifying some new default arguments (that can be overridden by command line args). This can be a list of config files ({'/'.join(supported_exts)}) with whatever arguments are set in them in order of lowest to highest priority. Usage: --config-file test.json test2.json.",
    )
    return parser


def get_args(
    parser: argparse.ArgumentParser,
    configs_root: str = "",
    default_ext: str = "json",
) -> argparse.Namespace:
    """Gets the arguments from an argparse parser with the added functionality of being able to use a config file. The parser load arguments in the following priority (for each argument it will go down the list until a value is found):
           - Command line arguments
           - Config files ordered from first listed to last listed
           - Parser default values

    Args:
        parser (argparse.ArgumentParser): The argument parser to use (will be modified, do not use after this method)
        configs_root (str, optional): The root to use for the config files. Defaults to "".
        default_ext (str, optional): The default extension to use if none is specified. Defaults to "json".

    Returns:
        argparse.Namespace: returns the parsed arguments.
    """

    parser = add_config_file_arg(parser=parser)

    args = parser.parse_args()

    args_dict = vars(args)
    cfg_files = list(reversed(args.config_file))

    if configs_root.endswith("/"):
        configs_root = configs_root[:-1]

    for cfg_file in cfg_files:
        cfg_file_name = cfg_file
        if not any(cfg_file_name.endswith(f".{ext}") for ext in supported_exts):
            cfg_file_name += f".{default_ext}"
        with open(f"{configs_root}/{cfg_file_name}") as f:
            ext = cfg_file_name.split(".")[-1]
            if ext not in supported_exts:
                raise NotImplementedError(f"Extension {ext} is not supported")
            configs_dict = supported_exts_loaders[ext](f)
            args_dict.update(configs_dict)

    parser = __clear_defaults(parser)

    set_args_dict = vars(parser.parse_args())
    args_dict.update(set_args_dict)

    return args


def print_markdown(parser: argparse.ArgumentParser, name: str, config_file=True, additional_info=None) -> None:    
    """Prints documentation for a given argument parser in the markdown format.

    Args:
        parser (argparse.ArgumentParser): The parser to generate documentation for.
        name (str): The title of the document to create.
        config_file (bool, optional): Whether or not to add the config file argument. Defaults to True.
        additional_info (str, optional): Any additional info to print before printing argparse info. Defaults to None.
    """
    if config_file:
        parser = add_config_file_arg(parser=parser)

    print(f"# {name}")
    print(parser.description)

    if additional_info is not None:
        print(additional_info)

    short_len = 10
    long_len = 15
    dest_len = 15
    default_len = 20
    type_len = 15
    help_len = 120

    print(f"\n\n## Arguments\n### Reference Table")
    print(
        f"|{'Short':{short_len}s}|{'Long':{long_len}s}|{'Config File Key':{dest_len}s}|{'Default':{default_len}s}|{'Type':{type_len}s}|{'Help':{help_len}s}|"
    )
    print(
        f"|{'-'*short_len}|{'-'*long_len}|{'-'*dest_len}|{'-'*default_len}|{'-'*type_len}|{'-'*help_len}"
    )

    custom_str = {"type": lambda c: c.__name__, "ABCMeta": lambda c: c.__name__}

    for act in parser._actions:
        print(
            f"|{act.option_strings[0]:{short_len}s}|{act.option_strings[1]:{long_len}s}|{act.dest:{dest_len}s}|{custom_str.get(type(act.default).__name__, str)(act.default):{default_len}s}|{('None' if act.type is None else act.type.__name__):{type_len}s}|{act.help:{help_len}s}|"
        )