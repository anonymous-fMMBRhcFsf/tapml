"""FFI APIs for tapml.serve"""
import tvm._ffi

# Exports functions registered via TVM_REGISTER_GLOBAL with the "tapml.serve" prefix.
# e.g. TVM_REGISTER_GLOBAL("tapml.serve.TextData")
tvm._ffi._init_api("tapml.serve", __name__)  # pylint: disable=protected-access
