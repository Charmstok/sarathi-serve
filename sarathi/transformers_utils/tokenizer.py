from typing import List, Optional, Tuple, Union

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from sarathi.logger import init_logger

logger = init_logger(__name__)

_FALLBACK_UNK_TOKEN = "[UNK]"
_QWEN3_TOKENIZER_FALLBACK = "Qwen/Qwen3-8B"


def _sanitize_tokens(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    tokens: List[object],
) -> List[str]:
    """Ensure all tokens are strings (fast tokenizers crash on None values)."""
    unk = getattr(tokenizer, "unk_token", None) or _FALLBACK_UNK_TOKEN
    sanitized: List[str] = []
    replaced = 0
    for token in tokens:
        if isinstance(token, str):
            sanitized.append(token)
        else:
            sanitized.append(unk)
            replaced += 1
    if replaced:
        logger.warning(
            f"Tokenizer produced {replaced} non-string tokens; replaced with {unk!r}."
        )
    return sanitized


def _load_tokenizer_candidate(
    source: str,
    *args,
    local_files_only: bool,
    trust_remote_code: bool,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    base_kwargs = dict(kwargs)
    base_kwargs["trust_remote_code"] = trust_remote_code
    return AutoTokenizer.from_pretrained(
        source,
        *args,
        local_files_only=local_files_only,
        **base_kwargs,
    )


def _is_tokenizer_usable(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> bool:
    try:
        probe_ids = tokenizer("hello", add_special_tokens=False)["input_ids"]
    except Exception:
        return False
    return bool(probe_ids) and getattr(tokenizer, "vocab_size", 0) > 1


def _get_family_tokenizer_fallbacks(tokenizer_name: str) -> List[str]:
    if (
        tokenizer_name.startswith("Qwen/Qwen3-")
        and tokenizer_name != _QWEN3_TOKENIZER_FALLBACK
    ):
        return [_QWEN3_TOKENIZER_FALLBACK]
    return []


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    tokenizer_path = tokenizer_name
    revision = kwargs.get("revision")
    try:
        tokenizer_path = snapshot_download(
            tokenizer_name,
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        pass

    candidates = [
        (tokenizer_path, True),
        (tokenizer_name, True),
    ]
    for fallback_name in _get_family_tokenizer_fallbacks(tokenizer_name):
        candidates.append((fallback_name, True))
        candidates.append((fallback_name, False))
    candidates.append((tokenizer_name, False))

    errors = []
    for source, local_only in candidates:
        try:
            tokenizer = _load_tokenizer_candidate(
                source,
                *args,
                local_files_only=local_only,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        except OSError as e:
            errors.append(f"{source} (local_only={local_only}): {e}")
            continue
        except TypeError as e:
            err_msg = "Failed to load the tokenizer."
            raise RuntimeError(err_msg) from e
        except ValueError as e:
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer is a custom "
                    "tokenizer not yet available in the HuggingFace transformers "
                    "library, consider setting `trust_remote_code=True` in LLM "
                    "or using the `--trust-remote-code` flag in the CLI."
                )
                raise RuntimeError(err_msg) from e
            errors.append(f"{source} (local_only={local_only}): {e}")
            continue

        if _is_tokenizer_usable(tokenizer):
            if source != tokenizer_name:
                logger.warning(
                    "Using tokenizer from %s for %s because the original tokenizer resources are incomplete.",
                    source,
                    tokenizer_name,
                )
            if not isinstance(tokenizer, PreTrainedTokenizerFast):
                logger.warning(
                    "Using a slow tokenizer. This might cause a significant "
                    "slowdown. Consider using a fast tokenizer instead."
                )
            return tokenizer

        errors.append(
            f"{source} (local_only={local_only}): unusable tokenizer "
            f"vocab_size={getattr(tokenizer, 'vocab_size', None)}"
        )

    raise RuntimeError(
        "Failed to load a usable tokenizer for "
        f"{tokenizer_name}. Tried: {'; '.join(errors)}"
    )


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def _convert_tokens_to_string_with_added_encoders(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    output_tokens: List[str],
    skip_special_tokens: bool,
) -> str:
    # Adapted from
    # https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # NOTE(woosuk): The following code is slow because it runs a for loop over
    # the output_tokens. In Python, running a for loop over a list can be slow
    # even when the loop body is very simple.
    sub_texts = []
    current_sub_text = []
    all_special_tokens = set(tokenizer.all_special_tokens)
    for token in output_tokens:
        if skip_special_tokens and token in all_special_tokens:
            continue
        if token in tokenizer.get_added_vocab():
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    return " ".join(sub_texts)


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    all_input_ids: List[int],
    prev_tokens: Optional[List[str]],
    prefix_offset: int = 0,
    read_offset: int = 0,
    skip_special_tokens: bool = False,
) -> Tuple[List[str], str, int, int]:
    new_token_id = all_input_ids[-1]

    # This is the first iteration for this sequence
    if prev_tokens is None:
        try:
            new_tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids[-6:], skip_special_tokens=skip_special_tokens
            )
        except (ValueError, OverflowError, TypeError) as e:
            new_tokens = ["[UNK]"] * 6
            logger.warning(f"Warning: {e}")

        output_tokens = new_tokens
        # 5 is an arbitrary value that should work for all
        # tokenizers (bigger = more conservative).
        # Subtract 1 extra to account for the generated token.
        prefix_offset = max(len(output_tokens) - 6, 0)
        read_offset = max(len(output_tokens) - 1, 0)
    else:
        # Put new_token_id in a list so skip_special_tokens is respected
        try:
            new_tokens = tokenizer.convert_ids_to_tokens(
                [new_token_id], skip_special_tokens=skip_special_tokens
            )
        except (ValueError, OverflowError, TypeError) as e:
            new_tokens = [prev_tokens[-1]]
            logger.warning(f"Warning: {e}")
        output_tokens = prev_tokens + new_tokens

    new_tokens = _sanitize_tokens(tokenizer, new_tokens)
    output_tokens = _sanitize_tokens(tokenizer, output_tokens)

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(
            output_tokens[prefix_offset:read_offset]
        )
        new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
        )

    if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        new_text = new_text[len(prefix_text) :]
        return new_tokens, new_text, read_offset, len(output_tokens)
    else:
        return new_tokens, "", prefix_offset, read_offset
