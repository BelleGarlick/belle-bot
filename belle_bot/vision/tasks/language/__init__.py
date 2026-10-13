import torch
from datasets import interleave_datasets, get_dataset_config_names, load_dataset
from torch.utils.data import DataLoader

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from . import train, model
from .train import collate_fn, tokenizer
from ..utils.models import TaskDefinition
from belle_bot.vision.tasks.utils.data import get_infinite_dataloader


def get_task_definition(config: VisionEncoderTrainingConfig, device: torch.device):
    """
    This function is used to get task definition for training the language
    projector task of the visual cortex

    :param config: The training config
    :param device: The device being used
    :return: The decoder config
    """
    mdl = model.VisualCortexLanguageProjection(8, 1024, 1024).to(device)

    # Load checkpoint and freeze model
    if config.model.language.file_path:
        mdl.load_state_dict(torch.load(config.model.language.file_path, map_location=device))

    language_dataset_name = "nvidia/Nemotron-Image-Training-v3"
    def load_language_dataset(split, sub_datasets):
        return interleave_datasets([
            load_dataset(language_dataset_name, sub, split=split, streaming=True)
            for sub in sub_datasets
        ], probabilities=[1.0 / len(sub_datasets)] * len(sub_datasets))

    language_sub_datasets = get_dataset_config_names(language_dataset_name)[0:10]

    language_train_dataset = load_language_dataset("train", language_sub_datasets)
    language_test_dataset = load_language_dataset("test", language_sub_datasets)

    language_train_loader = get_infinite_dataloader(DataLoader(
        language_train_dataset,
        batch_size=1,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    ))
    language_test_loader = get_infinite_dataloader(DataLoader(
        language_test_dataset,
        batch_size=1,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    ))

    return TaskDefinition(
        model=mdl,
        perform_step=train.perform_step,
        train_set=language_train_loader,
        test_set=language_test_loader
    )
