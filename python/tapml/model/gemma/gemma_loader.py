"""
This file specifies how TapML's Gemma parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from tapml.loader import ExternMapping
from tapml.quantization import Quantization

from .gemma_model import GemmaConfig, GemmaForCausalLM


def huggingface(model_config: GemmaConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of TapML parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : GemmaConfig
        The configuration of the Gemma model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from TapML to HuggingFace PyTorch.
    """
    model = GemmaForCausalLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"model.layers.{i}.self_attn"
        tapml_name = f"{attn}.qkv_proj.weight"
        tapml_param = named_parameters[tapml_name]
        mapping.add_mapping(
            tapml_name,
            [
                f"{attn}.q_proj.weight",
                f"{attn}.k_proj.weight",
                f"{attn}.v_proj.weight",
            ],
            functools.partial(
                lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                dtype=tapml_param.dtype,
            ),
        )
        # Add gates in MLP
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
        # Modify RMS layernorm weights, since Gemma model adds 1 to the weights
        # We add 1 to the weights here for efficiency purpose
        tapml_name = f"model.layers.{i}.input_layernorm.weight"
        tapml_param = named_parameters[tapml_name]
        mapping.add_mapping(
            tapml_name,
            [tapml_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[tapml_name].dtype,
            ),
        )

        tapml_name = f"model.layers.{i}.post_attention_layernorm.weight"
        tapml_param = named_parameters[tapml_name]
        mapping.add_mapping(
            tapml_name,
            [tapml_name],
            functools.partial(
                lambda x, dtype: (x + 1).astype(dtype),
                dtype=named_parameters[tapml_name].dtype,
            ),
        )

    tapml_name = "model.norm.weight"
    tapml_param = named_parameters[tapml_name]
    mapping.add_mapping(
        tapml_name,
        [tapml_name],
        functools.partial(
            lambda x, dtype: (x + 1).astype(dtype),
            dtype=named_parameters[tapml_name].dtype,
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
