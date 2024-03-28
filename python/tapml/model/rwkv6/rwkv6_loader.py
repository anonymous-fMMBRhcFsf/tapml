"""
This file specifies how TapML's RWKV6 parameter maps from other formats, for example HuggingFace
PyTorch, HuggingFace safetensors.
"""

import functools

from ...loader import ExternMapping
from ...quantization import Quantization
from .rwkv6_model import RWKV6_ForCasualLM, RWKV6Config


def huggingface(model_config: RWKV6Config, quantization: Quantization) -> ExternMapping:
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
    model = RWKV6_ForCasualLM(model_config)
    if quantization is not None:
        model.to(quantization.model_dtype)
    _, _named_params = model.export_tvm(  # pylint: disable=unbalanced-tuple-unpacking
        spec=model.get_default_spec()
    )
    named_parameters = dict(_named_params)

    mapping = ExternMapping()

    for i in range(model_config.num_hidden_layers):
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
