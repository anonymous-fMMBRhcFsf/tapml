"""
This file specifies how TapML's BaichuanLM parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from tapml.loader import ExternMapping
from tapml.quantization import Quantization

from .baichuan_model import BaichuanConfig, BaichuanForCausalLM


def huggingface(model_config: BaichuanConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of TapML parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : BaichuanConfig
        The configuration of the Baichuan model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from TapML to HuggingFace PyTorch.
    """
    model = BaichuanForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        mlp = f"model.layers.{i}.mlp"
        tapml_name = f"{mlp}.gate_up_proj.weight"
        tapml_param = named_parameters[tapml_name]
        mapping.add_mapping(
            tapml_name,
            [
                f"{mlp}.gate_proj.weight",
                f"{mlp}.up_proj.weight",
            ],
            functools.partial(
                lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                dtype=tapml_param.dtype,
            ),
        )

    for tapml_name, tapml_param in named_parameters.items():
        if tapml_name not in mapping.param_map:
            mapping.add_mapping(
                tapml_name,
                [tapml_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=tapml_param.dtype,
                ),
            )
    return mapping
