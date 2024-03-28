"""FFI APIs for tapml"""
import tvm._ffi

# Exports functions registered via TVM_REGISTER_GLOBAL with the "tapml" prefix.
# e.g. TVM_REGISTER_GLOBAL("tapml.Tokenizer")
tvm._ffi._init_api("tapml", __name__)  # pylint: disable=protected-access
