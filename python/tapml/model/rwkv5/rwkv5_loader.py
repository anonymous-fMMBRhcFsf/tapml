"""
This file specifies how TapML's RWKV5 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from ...loader import ExternMapping
from ...quantization import Quantization
from .rwkv5_model import RWKV5_ForCasualLM, RWKV5Config


def huggingface(model_config: RWKV5Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of TapML parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : RWKVConfig
        The configuration of the Mistral model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from TapML to HuggingFace PyTorch.
    """
    model = RWKV5_ForCasualLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(  # pylint: disable=unbalanced-tuple-unpacking
        spec=model.get_default_spec()
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # convert time_decay
        tapml_name = f"model.blocks.{i}.attention.time_decay"
        hf_name = f"rwkv.blocks.{i}.attention.time_decay"
        tapml_param = named_parameters[tapml_name]
        if tapml_param.dtype != "float32":
            raise ValueError(f"RWKV5 time_decay should be float32, got {tapml_param.dtype}")
        mapping.add_mapping(
            tapml_name,
            [hf_name],
            functools.partial(
                lambda x, dtype: np.exp(-np.exp(x.astype(dtype))),
                dtype=tapml_param.dtype,
            ),
        )

        # rescale
        if model_config.rescale_every > 0:
            for name in ["feed_forward.value.weight", "attention.output.weight"]:
                tapml_name = f"model.blocks.{i}.{name}"
                hf_name = f"rwkv.blocks.{i}.{name}"
                tapml_param = named_parameters[tapml_name]

                mapping.add_mapping(
                    tapml_name,
                    [hf_name],
                    functools.partial(
                        lambda x, dtype, t: x.astype(dtype) / (2**t),
                        dtype=tapml_param.dtype,
                        t=i // model_config.rescale_every,
                    ),
                )

    for tapml_name, tapml_param in named_parameters.items():
        if tapml_name not in mapping.param_map:
            hf_name = tapml_name.replace("model", "rwkv")
            mapping.add_mapping(
                tapml_name,
                [hf_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=tapml_param.dtype,
                ),
            )

    return mapping
