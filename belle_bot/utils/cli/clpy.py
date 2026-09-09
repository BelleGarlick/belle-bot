import sys
from typing import Type, Union

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass


def print_help(config: Type[BaseModel], prefix=" --"):
    for field, value in config.model_fields.items():
        if isinstance(value.annotation, ModelMetaclass):
            print_help(value.annotation, prefix=prefix + f"{field}.")
        else:
            text = f"{prefix}{field}: {value.description or 'No description provided'}"
            if value.default is not None:
                text = f"{text} (default: {value.default})"
            print(text)


def print_values(config: BaseModel, prefix=" - "):
    if not isinstance(config, dict):
        print_values(config.model_dump())
        return

    for key, value in config.items():
        if isinstance(value, dict):
            print(prefix + f"{key}")
            print_values(value, prefix="    " + prefix)

        else:
            print(prefix + f"{key}: {value}")


def to_dict(config: Union[BaseModel, dict]) -> dict:
    if isinstance(config, BaseModel):
        config = config.model_dump()

    def flatten(d, prefix=""):
        items = []
        for k, v in d.items():
            new_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten(config)


def parse_cli_args(default_args):
    print(sys.argv)

    args_dict = {}
    for key, item in to_dict(default_args).items():
        args_dict["--" + key] = item

    for i in range(len(sys.argv) - 1):
        if i == "-h" or i == "--help":
            print_help(args_dict)
            sys.exit()

        if sys.argv[i] in args_dict:
            args_dict[sys.argv[i]] = sys.argv[i + 1]
            # todo update the args
            # todo somehow parse

    return default_args