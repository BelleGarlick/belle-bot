import sys
from typing import Type, Union

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass


def print_help(config: Type[BaseModel], prefix=" --"):
    model_schema = config.model_json_schema()

    bold = "\033[1m"
    header = "\033[95m"
    end = "\033[0m"

    for key, value in model_schema['properties'].items():
        if '$ref' in value:
            # todo iterate through the model schema again
            # breakpoint()
            pass
        print(f"{bold}{header}--{key}{end}: {value.get('description', 'no description')}")
        print(f"  default: {value.get('default', None)}")
        # print(key, value)

    # breakpoint()


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


def parse_cli_args[T](default_args: T) -> T:
    print(sys.argv)

    args_dict = {}
    for key, item in to_dict(default_args).items():
        args_dict["--" + key] = item

    # todo update this whole method its stupid
    for i in range(len(sys.argv)):
        if sys.argv[i] == "-h" or sys.argv[i] == "--help":
            print_help(default_args)
            sys.exit()

        key = sys.argv[i]
        if key in args_dict and sys.argv[i].startswith("--"):
            args_dict[key] = sys.argv[i + 1]
            # todo update the args
            # todo somehow parse

    # todo, do this above so we dont need to update the args dict
    for key, value in args_dict.items():
        key = key.replace("--", "")
        tokens = key.split(".")
        current_arg_level = default_args
        for i in range(len(tokens) - 1):
            current_arg_level = getattr(current_arg_level, tokens[i])
        setattr(current_arg_level, tokens[-1], value)

    return default_args