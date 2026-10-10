import torch
from datasets import interleave_datasets, get_dataset_config_names, load_dataset
from torch.utils.data import DataLoader

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from . import train, model
from .train import collate_fn, tokenizer
from ..utils.models import TaskDefinition


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
    language_sub_datasets = get_dataset_config_names(language_dataset_name)[0:1]
    language_dataset = interleave_datasets([
        load_dataset(language_dataset_name, sub, split="train", streaming=True)
        for sub in language_sub_datasets
    ], probabilities=[1.0 / len(language_sub_datasets)] * len(language_sub_datasets))

    language_train_loader = iter(DataLoader(
        language_dataset,
        batch_size=1,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    ))
    language_test_loader = iter(DataLoader(
        language_dataset,
        batch_size=1,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    ))

    return TaskDefinition(
        model=mdl,
        perform_step=train.perform_step,
        train_set=language_train_loader,
        test_set=language_test_loader
    )
