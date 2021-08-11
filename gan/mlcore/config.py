import os
import sys
import copy

import argparse
import logging
import json

from mlcore.utils import flat_dict, unflat_dict, get_element


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class ConfigEntry:
    def __init__(self, *args, default=None, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.default = default
        if default is not None:
            self.kwargs.update({"default": default})

    def argparser_args(self):
        return self.args

    def argparser_kwargs(self):
        return self.kwargs


global_config = {}


def config_add_options(name=None):
    if name is not None:

        def handle(func):
            assert name not in global_config, f"Config: {name} is already defined"
            global_config[name] = {}
            for k, v in func().items():
                global_config[name][k] = v

        return handle

    def handle(func):

        for k, v in func().items():
            assert k not in global_config, f"Config: {name} is already defined"
            global_config[k] = v

    return handle


class ArgsObject(argparse.Namespace):
    def __init__(self, data_dict):
        self.__data = data_dict

        for key, value in data_dict.items():
            if isinstance(value, dict):
                value = ArgsObject(value)
            setattr(self, key, value)

    def __str__(self):
        return json.dumps(self.__data, indent=2)

    def __contains__(self, key):
        return key in self.__data

    def __getitem__(self, item):
        obj = getattr(self, item)
        if isinstance(obj, Config):
            obj = obj.__data
        return obj

    def keys(self):
        return self.__data.keys()


class Config:
    def __init__(self):

        self._default_config_args = self.default()
        self._module_config_args = global_config

        self._flat_config_from_defaults = self.build_flat_config_dict(self._module_config_args)

        self._flat_config_from_args = self.parse_args({**self._default_config_args, **self._module_config_args})

        self._flat_config_from_file = {}

        config_file = self._flat_config_from_args.get("config", None)

        if config_file is None:
            config_file = self._flat_config_from_defaults.get("config", None)

        if config_file is not None and os.path.isfile(config_file):
            self._flat_config_from_file = self.parse_file(config_file)

        # Set parameter from config file as new default values
        if config_file is not None and os.path.isfile(config_file):
            self._flat_config_from_args = self.parse_args(
                {**self._default_config_args, **self._module_config_args}, defaults=self._flat_config_from_file
            )

        self.config = unflat_dict(
            {**self._flat_config_from_defaults, **self._flat_config_from_file, **self._flat_config_from_args}
        )
        config_dump_file = self.config.get("config_dump", None)
        if config_dump_file is not None:
            with open(config_dump_file, "w") as f:
                json.dump(self.config, f, indent=2)
        # flat_config.update({k: getattr(args, k) for k in vars(args) if getattr(args, k) is not None})

    def parse_file(self, path):

        with open(path, "r") as f:
            return self.build_flat_config_dict(json.load(f))

    def parse_args(self, args_list, defaults=None):

        args_list = flat_dict(args_list)

        parser = argparse.ArgumentParser(description="")
        for k, v in args_list.items():
            if isinstance(v, ConfigEntry):
                parser.add_argument(*v.argparser_args(), f"--{k}", **v.argparser_kwargs())
            if isinstance(v, bool):
                parser.add_argument(f"--{k}", type=str2bool)
            elif isinstance(v, (set, list)):
                parser.add_argument(f"--{k}", nargs="+")
            elif isinstance(v, int):
                parser.add_argument(f"--{k}", type=int)
            elif isinstance(v, float):
                parser.add_argument(f"--{k}", type=float)

        if defaults is not None:
            parser.set_defaults(**defaults)

        args = parser.parse_args()

        # return {k: getattr(args, k) for k in vars(args) if getattr(args, k) is not None}
        return {k: getattr(args, k) for k in vars(args)}

    def default(self):
        return {
            "verbose": ConfigEntry("-v", default=None, const=True, action="store_const", help="Verbose output"),
            "debug": ConfigEntry("-vv", default=None, const=True, action="store_const", help="Debug output"),
            "config_dump": ConfigEntry(type=str, help="Debug output"),
            "config": ConfigEntry("-c", help="Config file"),
        }

    def build_flat_config_dict(self, config_args):
        flatted_dict = flat_dict(copy.deepcopy(config_args))

        def get_value(x):
            return x.default if isinstance(x, ConfigEntry) else x

        return {k: get_value(v) for k, v in flatted_dict.items()}

    def to_args(self):

        o = ArgsObject(copy.deepcopy(self.config))

        return o

    def to_dict(self):

        return copy.deepcopy(self.config)

    def to_flat_args(self):
        o = ArgsObject(flat_dict(copy.deepcopy(self.config)))
        return o

    def __repr__(self):
        return str(
            unflat_dict(
                {**self._flat_config_from_defaults, **self._flat_config_from_file, **self._flat_config_from_args}
            )
        )
