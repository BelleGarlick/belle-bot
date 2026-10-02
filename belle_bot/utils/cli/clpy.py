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


def parse_cli_args[T: BaseModel](default_args: T) -> T:
    print(sys.argv)

    # 1. Flatten the existing config to know what keys we expect
    flat_config = to_dict(default_args)
    
    # 2. Extract overrides from sys.argv
    overrides = {}
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "-h" or arg == "--help":
            print_help(default_args.__class__)
            sys.exit()
        
        if arg.startswith("--"):
            key = arg[2:]
            if key in flat_config:
                if i + 1 < len(sys.argv):
                    overrides[key] = sys.argv[i+1]
                    i += 1
        i += 1

    # 3. Create a nested dictionary from the flattened overrides
    def unflatten(d):
        result = {}
        for key, value in d.items():
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        return result

    nested_overrides = unflatten(overrides)

    # 4. Update the default model dump with overrides
    def deep_update(mapping, *updating_mappings):
        updated_mapping = mapping.copy()
        for updating_mapping in updating_mappings:
            for k, v in updating_mapping.items():
                if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                    updated_mapping[k] = deep_update(updated_mapping[k], v)
                else:
                    updated_mapping[k] = v
        return updated_mapping

    updated_dict = deep_update(default_args.model_dump(), nested_overrides)

    # 5. Create a new instance of the model with the updated dictionary to trigger validation/casting
    return default_args.__class__.model_validate(updated_dict)