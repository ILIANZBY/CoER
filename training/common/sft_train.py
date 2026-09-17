"""Shared all-assistant-turn SFT training for attacker and defender."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from sft_data import ConversationSFTDataset

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model"})


@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "Path to SFT JSONL data"})
    max_seq_length: int = field(default=32768, metadata={"help": "Max sequence length"})
    expected_examples: int = field(default=0, metadata={"help": "Expected corpus size; 0 skips the check"})
    save_final_model: bool = field(
        default=True,
        metadata={"help": "Save the consolidated model and trainer state after training"},
    )


class HdfsCompatibleTrainer(Trainer):
    """Trainer that writes model weights with ``torch.save`` instead of safetensors.

    Some shared FUSE mounts do not implement the filesystem call
    used by safetensors' serializer. Transformers 5.x always uses safetensors
    and no longer exposes ``TrainingArguments.save_safetensors``, so saving the
    state dict explicitly is required here.
    """

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving PyTorch-format model checkpoint to %s", output_dir)

        if state_dict is None:
            state_dict = self.model.state_dict()
        torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

        model_to_save = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        if getattr(model_to_save, "config", None) is not None:
            model_to_save.config.save_pretrained(output_dir)
        if getattr(model_to_save, "generation_config", None) is not None:
            model_to_save.generation_config.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        level=logging.INFO if training_args.should_log else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if training_args.resume_from_checkpoint:
        raise ValueError("Canonical SFT starts fresh; pass the initialization via model_name_or_path")
    output = Path(training_args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; existing experiments will not be overwritten")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    # The local checkpoint uses <|endoftext|> as its config EOS while the chat
    # template terminates assistant turns with <|im_end|>. Stop on either token
    # and persist the same setting with the saved model.
    configured_eos = model.generation_config.eos_token_id
    if configured_eos is None:
        configured_eos_ids = []
    elif isinstance(configured_eos, int):
        configured_eos_ids = [configured_eos]
    else:
        configured_eos_ids = list(configured_eos)
    eos_token_ids = list(dict.fromkeys([*configured_eos_ids, tokenizer.eos_token_id]))
    model.generation_config.eos_token_id = eos_token_ids
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    text_config = getattr(model.config, "text_config", model.config)
    model.config.use_cache = True
    text_config.use_cache = True
    text_config.eos_token_id = eos_token_ids
    text_config.pad_token_id = tokenizer.pad_token_id
    model_max_length = getattr(text_config, "max_position_embeddings", None)
    if model_max_length is not None and data_args.max_seq_length > model_max_length:
        raise ValueError(f"max_seq_length={data_args.max_seq_length} exceeds the model limit of {model_max_length}")

    # Ensure model vocab covers all token IDs from tokenizer
    # Model config has vocab_size=248320, tokenizer max_id=248076, keep model's larger size
    if len(tokenizer) > text_config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load dataset
    train_dataset = ConversationSFTDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )

    if data_args.expected_examples and len(train_dataset) != data_args.expected_examples:
        raise ValueError(f"Expected {data_args.expected_examples} examples, found {len(train_dataset)}")
    if not len(train_dataset):
        raise ValueError("SFT corpus is empty")

    # Data collator with padding
    def data_collator(features):
        features = [{key: torch.tensor(value, dtype=torch.long) for key, value in item.items()} for item in features]
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch["input_ids"].append(
                torch.cat([f["input_ids"], torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
            )
            batch["attention_mask"].append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
            batch["labels"].append(torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
        batch = {k: torch.stack(v) for k, v in batch.items()}

        # Qwen3.5 accepts token positions in logits_to_keep. Computing the
        # 248K-way vocabulary logits only where the following label is
        # supervised makes long, fully preserved trajectories practical. The
        # supplied shift_labels keeps the loss identical to the standard causal
        # LM shift. Positions are shared across the local batch; labels remain
        # masked independently for each example.
        target_mask = batch["labels"][:, 1:].ne(-100)
        logits_to_keep = target_mask.any(dim=0).nonzero(as_tuple=False).flatten()
        if logits_to_keep.numel() == 0:
            raise ValueError("Batch has no supervised next-token targets")
        batch["logits_to_keep"] = logits_to_keep
        batch["shift_labels"] = batch["labels"][:, logits_to_keep + 1]
        # Keep deployable/checkpoint configs cache-enabled while disabling KV
        # cache only for training forwards with gradient checkpointing.
        batch["use_cache"] = False
        return batch

    # Trainer (use processing_class for transformers 5.x)
    trainer = HdfsCompatibleTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    if data_args.save_final_model:
        trainer.save_model()
        trainer.save_state()
        logger.info("Training complete. Model saved to %s", training_args.output_dir)
    else:
        logger.info("Training complete; final model save disabled")


if __name__ == "__main__":
    main()
