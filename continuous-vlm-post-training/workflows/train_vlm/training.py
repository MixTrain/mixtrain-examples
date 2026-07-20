"""Prepare VLM data and run distributed TRL training."""

import json
import re
from pathlib import Path


def prepare_splits(data, destination: Path) -> int:
    """Write train and validation splits in TRL's vision SFT format."""
    from datasets import DatasetDict, Image, Sequence

    dataset = data.load_files(to="path").to_huggingface()

    dataset = dataset.map(
        _as_conversation,
        remove_columns=dataset.column_names,
    ).cast_column("images", Sequence(Image()))
    validation_size = max(1, len(dataset) // 5)
    split = dataset.train_test_split(test_size=validation_size, seed=42)
    DatasetDict(train=split["train"], validation=split["test"]).save_to_disk(
        destination
    )
    return len(dataset)


def _as_conversation(row: dict) -> dict:
    return {
        "images": [row["image"]],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["ground_truth"]}],
            },
        ],
    }


def _train_worker(
    local_rank: int,
    gpu_per_node: int,
    dataset_path: str,
    output_dir: str,
    base_model: str,
    num_epochs: int,
    learning_rate: float,
) -> None:
    """Run one SFTTrainer process."""
    import os

    node_rank = int(os.environ.get("RANK", "0"))
    num_nodes = int(os.environ.get("WORLD_SIZE", "1"))
    os.environ["RANK"] = str(node_rank * gpu_per_node + local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(num_nodes * gpu_per_node)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    torch.cuda.set_device(local_rank)
    splits = load_from_disk(dataset_path)
    processor = AutoProcessor.from_pretrained(base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
    )

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        bf16=True,
        max_length=None,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        save_only_model=True,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        seed=42,
    )
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()

    if os.environ["RANK"] == "0":
        trainer.state.save_to_json(str(Path(output_dir) / "trainer_state.json"))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def run_distributed_training(
    dataset_path: str,
    output_dir: str,
    base_model: str,
    num_epochs: int,
    learning_rate: float,
) -> list[Path]:
    """Spawn one training process per GPU and return the epoch checkpoints."""
    import os

    import torch
    import torch.multiprocessing as mp

    gpu_per_node = torch.cuda.device_count()

    mp.spawn(
        _train_worker,
        args=(
            gpu_per_node,
            dataset_path,
            output_dir,
            base_model,
            num_epochs,
            learning_rate,
        ),
        nprocs=gpu_per_node,
        join=True,
    )

    return sorted(Path(output_dir).glob("checkpoint-*"), key=checkpoint_step)


def load_log_history(output_dir: str) -> list[dict]:
    state = json.loads((Path(output_dir) / "trainer_state.json").read_text())
    return state["log_history"]


def epoch_metrics(history: list[dict], epoch: int) -> tuple[float | None, float | None]:
    """Return the (training_loss, validation_loss) logged for one epoch."""
    train_loss = None
    validation_loss = None
    for event in history:
        event_epoch = event.get("epoch")
        if event_epoch is None or round(float(event_epoch)) != epoch:
            continue
        if "loss" in event:
            train_loss = float(event["loss"])
        if "eval_loss" in event:
            validation_loss = float(event["eval_loss"])
    return train_loss, validation_loss


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Unexpected checkpoint directory: {path.name}")
    return int(match.group(1))
