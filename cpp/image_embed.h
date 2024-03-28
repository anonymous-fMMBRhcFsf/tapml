/*!
 *  Copyright (c) 2023 by Contributors
 * \file image_embed.h
 * \brief Implementation of image embedding pipeline.
 */
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/module.h>

#include "base.h"

namespace tapml {
namespace llm {

// explicit export via TVM_DLL
TapML_DLL tvm::runtime::Module CreateImageModule(DLDevice device);

}  // namespace llm
}  // namespace tapml
