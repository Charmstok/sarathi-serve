from typing import Optional

from huggingface_hub import snapshot_download
from transformers import AutoConfig, PretrainedConfig

from sarathi.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "qwen": QWenConfig,
    "RefinedWeb": RWConfig,  # For tiiuae/falcon-40b(-instruct)
    "RefinedWebModel": RWConfig,  # For tiiuae/falcon-7b(-instruct)
    "yi": YiConfig,
}


def get_config(
    model: str, trust_remote_code: bool, revision: Optional[str] = None
) -> PretrainedConfig:
    model_path = model
    try:
        model_path = snapshot_download(
            model,
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        pass

    load_kwargs = {
        "trust_remote_code": trust_remote_code,
        "revision": revision,
    }
    try:
        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            **load_kwargs,
        )
    except OSError:
        config = AutoConfig.from_pretrained(model, **load_kwargs)
    except ValueError as e:
        if (
            not trust_remote_code
            and "requires you to execute the configuration file" in str(e)
        ):
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        else:
            raise e
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        try:
            config = config_class.from_pretrained(
                model_path,
                revision=revision,
                local_files_only=True,
            )
        except OSError:
            config = config_class.from_pretrained(model, revision=revision)
    return config
