"""
This file specifies how TapML's GPT-2 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""
import functools

from tapml.loader import ExternMapping
from tapml.quantization import Quantization

from .gpt2_model import GPT2Config, GPT2LMHeadModel


def huggingface(model_config: GPT2Config, quantization: Quantization) -> ExternMapping:
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
    model = GPT2LMHeadModel(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params, _ = model.export_tvm(  # type: ignore[misc]
        spec=model.get_default_spec(),
        allow_extern=True,
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    mapping.add_mapping(
        "lm_head.weight",
        ["wte.weight"],
        functools.partial(
            lambda x, dtype: x.astype(dtype),
            dtype=named_parameters["transformer.wte.weight"].dtype,
        ),
    )

    for i in range(model_config.n_layer):
        mapping.add_unused(f"h.{i}.attn.bias")

        # Transpose c_attn, c_proj and c_fc weights since GPT-2 uses Conv1D
        for conv1d_weight_name in ["attn.c_attn", "attn.c_proj", "mlp.c_proj", "mlp.c_fc"]:
            src_name = f"h.{i}.{conv1d_weight_name}.weight"
            tapml_name = f"transformer.{src_name}"
            mapping.add_mapping(
                tapml_name,
                [src_name],
                functools.partial(
                    lambda x, dtype: x.transpose().astype(dtype),
                    dtype=named_parameters[tapml_name].dtype,
                ),
            )

    for tapml_name, tapml_param in named_parameters.items():
        if tapml_name not in mapping.param_map:
            # transformer.h.0.attn.c_attn.weight --> h.0.attn.c_attn.weight
            source_name = tapml_name.split(".", 1)[1]
            mapping.add_mapping(
                tapml_name,
                [source_name],
                functools.partial(
                    lambda x, dtype: x.astype(dtype),
                    dtype=tapml_param.dtype,
                ),
            )

    return mapping
