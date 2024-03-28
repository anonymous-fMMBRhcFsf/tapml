"""
This file specifies how TapML's Llava parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

import numpy as np

from tapml.loader import ExternMapping
from tapml.quantization import Quantization

from .llava_model import LlavaConfig, LlavaForCasualLM
from .llava_quantization import awq_quant


def huggingface(model_config: LlavaConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of TapML parameters to
    the names of HuggingFace PyTorch parameters.

    Parameters
    ----------
    model_config : LlavaConfig
        The configuration of the Llava model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from TapML to HuggingFace PyTorch.
    """
    model = LlavaForCasualLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(), allow_extern=True
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.text_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"language_model.model.layers.{i}.self_attn"
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
        mlp = f"language_model.model.layers.{i}.mlp"
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
        # inv_freq is not used in the model
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")

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


def awq(model_config: LlavaConfig, quantization: Quantization) -> ExternMapping:
    """Returns a parameter mapping that maps from the names of TapML parameters to
    the names of AWQ parameters.
    Parameters
    ----------
    model_config : LlavaConfig
        The configuration of the Llava model.

    quantization : Quantization
        The quantization configuration.

    Returns
    -------
    param_map : ExternMapping
        The parameter mapping from TapML to AWQ.
    """
    model, _ = awq_quant(model_config, quantization)
    _, _named_params = model.export_tvm(spec=model.get_default_spec())
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.text_config.num_hidden_layers):
        # Add QKV in self attention
        attn = f"language_model.model.layers.{i}.self_attn"
        for quantize_suffix in ["qweight", "qzeros", "scales"]:
            tapml_name = f"{attn}.qkv_proj.{quantize_suffix}"
            assert tapml_name in named_parameters
            tapml_param = named_parameters[tapml_name]
            mapping.add_mapping(
                tapml_name,
                [
                    f"{attn}.q_proj.{quantize_suffix}",
                    f"{attn}.k_proj.{quantize_suffix}",
                    f"{attn}.v_proj.{quantize_suffix}",
                ],
                functools.partial(
                    lambda q, k, v, dtype: np.concatenate([q, k, v], axis=0).astype(dtype),
                    dtype=tapml_param.dtype,
                ),
            )

        # Concat gate and up in MLP
        mlp = f"language_model.model.layers.{i}.mlp"
        for quantize_suffix in ["qweight", "qzeros", "scales"]:
            tapml_name = f"{mlp}.gate_up_proj.{quantize_suffix}"
            assert tapml_name in named_parameters
            tapml_param = named_parameters[tapml_name]
            mapping.add_mapping(
                tapml_name,
                [
                    f"{mlp}.gate_proj.{quantize_suffix}",
                    f"{mlp}.up_proj.{quantize_suffix}",
                ],
                functools.partial(
                    lambda gate, up, dtype: np.concatenate([gate, up], axis=0).astype(dtype),
                    dtype=tapml_param.dtype,
                ),
            )

        # inv_freq is not used in the model
        mapping.add_unused(f"{attn}.rotary_emb.inv_freq")

    for tapml_name, tapml_param in named_parameters.items():
        if tapml_name not in mapping.param_map:
            mapping.add_mapping(
                tapml_name,
                [tapml_name],
                functools.partial(lambda x, dtype: x.astype(dtype), dtype=tapml_param.dtype),
            )
    return mapping
