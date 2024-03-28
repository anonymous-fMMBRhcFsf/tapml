"""
This file specifies how TapML's QWen2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from tapml.loader import ExternMapping
from tapml.quantization import Quantization

from .qwen2_model import QWen2Config, QWen2LMHeadModel


def huggingface(model_config: QWen2Config, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of TapML parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : GPT2Config
        The configuration of the GPT-2 model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from TapML to HuggingFace PyTorch.
    """
    model = QWen2LMHeadModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # map attention weight
        attn = f"model.layers.{i}.self_attn"
        for weight_type in ["weight", "bias"]:
            tapml_name = f"{attn}.c_attn.{weight_type}"
            tapml_param = named_parameters[tapml_name]
            mapping.add_mapping(
                tapml_name,
                [
                    f"{attn}.q_proj.{weight_type}",
                    f"{attn}.k_proj.{weight_type}",
                    f"{attn}.v_proj.{weight_type}",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=tapml_param.dtype,
                ),
            )
        # map mlp weight
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
