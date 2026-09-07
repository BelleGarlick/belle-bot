from typing import Type

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass


def print_help(config: Type[BaseModel], prefix=" --"):
    for field, value in config.model_fields.items():
        if isinstance(value.annotation, ModelMetaclass):
            print_help(value.annotation, prefix=prefix + f"{field}.")
        else:
            text = f"{prefix}{field}: {value.description or "No description provided"}"
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
