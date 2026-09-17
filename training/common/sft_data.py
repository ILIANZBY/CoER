"""Shared, dependency-free SFT token/label construction."""

import json
import logging

logger = logging.getLogger(__name__)


class ConversationSFTDataset:
    """Pre-serialized Qwen conversations; every assistant turn receives CE loss."""

    IGNORE_INDEX = -100
    SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}

    def __init__(self, data_path: str, tokenizer, max_seq_length: int):
        if max_seq_length < 2:
            raise ValueError("max_seq_length must be at least 2")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        # Resolve and validate the structural tokens once. Labels are not built
        # by scanning for these IDs, because message content may contain them.
        self.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.im_start_id is None or self.im_end_id is None:
            raise ValueError("Tokenizer must define <|im_start|> and <|im_end|>")
        if self.im_start_id == self.im_end_id:
            raise ValueError("<|im_start|> and <|im_end|> must have distinct token IDs")

        # Load data
        self.records = []
        with open(data_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if line.strip():
                    try:
                        self.records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSON in {data_path} at line {line_number}") from exc
        logger.info(f"Loaded {len(self.records)} records from {data_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        conversations = record["conversations"]
        input_ids, labels = self._build_tokens_and_labels(conversations, idx)

        if len(input_ids) > self.max_seq_length:
            raise ValueError(
                f"Record {idx} has {len(input_ids)} tokens, exceeding "
                f"max_seq_length={self.max_seq_length}; truncation is disabled"
            )

        attention_mask = [1] * len(input_ids)

        if not any(label != self.IGNORE_INDEX for label in labels):
            raise ValueError(
                f"Record {idx} has no supervised tokens after masking/truncation; "
                "check conversation content and max_seq_length"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_tokens_and_labels(self, conversations, record_idx: int):
        """Build ChatML blocks without re-rendering historical assistant turns.

        Qwen's stock template intentionally removes reasoning from historical
        assistant messages. SFT needs the original trajectory, so the structural
        ChatML tokens are emitted here while message content is kept verbatim
        (apart from the template-equivalent outer whitespace trim).
        """
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"Record {record_idx} has no conversation messages")

        input_ids: list[int] = []
        labels: list[int] = []
        supervised_assistant_turns = 0

        for message_idx, message in enumerate(conversations):
            role = message.get("role")
            content = message.get("content")
            if role not in self.SUPPORTED_ROLES:
                raise ValueError(f"Record {record_idx} message {message_idx} has unsupported role {role!r}")
            if not isinstance(content, str):
                raise ValueError(f"Record {record_idx} message {message_idx} content must be a string")

            if message.get("tool_calls"):
                raise ValueError("Serialize tool calls into content using the model protocol before SFT")
            # Qwen tool returns appear as user-side tool_response blocks.
            rendered_role = "user" if role == "tool" else role
            if role == "tool":
                content = "<tool_response>\n" + content.strip() + "\n</tool_response>"
            header_ids = self._encode(f"<|im_start|>{rendered_role}\n")
            content_ids = self._encode(content.strip())
            suffix_ids = self._encode("<|im_end|>\n")
            if not header_ids or header_ids[0] != self.im_start_id:
                raise ValueError("Tokenizer did not encode the ChatML start token as expected")
            if not suffix_ids or suffix_ids[0] != self.im_end_id:
                raise ValueError("Tokenizer did not encode the ChatML end token as expected")

            input_ids.extend(header_ids)
            input_ids.extend(content_ids)
            input_ids.extend(suffix_ids)
            labels.extend([self.IGNORE_INDEX] * len(header_ids))

            train_this_turn = role == "assistant"
            if train_this_turn:
                supervised_assistant_turns += 1
                labels.extend(content_ids)
                # Supervise the structural end token, but not the following newline.
                labels.extend([self.im_end_id])
                labels.extend([self.IGNORE_INDEX] * (len(suffix_ids) - 1))
            else:
                labels.extend([self.IGNORE_INDEX] * (len(content_ids) + len(suffix_ids)))

        if supervised_assistant_turns == 0:
            raise ValueError(f"Record {record_idx} has no assistant turn selected for training")
        if len(input_ids) != len(labels):
            raise RuntimeError("Internal error: input IDs and labels have different lengths")
        return input_ids, labels
