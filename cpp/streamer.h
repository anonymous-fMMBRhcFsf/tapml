/*!
 *  Copyright (c) 2023 by Contributors
 * \file streamer.h
 * \brief Header of streamers in TapML.
 */

#ifndef TAPML_STREAMER_H_
#define TAPML_STREAMER_H_

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/object.h>

#include "tokenizers.h"

namespace tapml {
namespace llm {

using namespace tvm::runtime;

/****************** TextStreamer ******************/

/*!
 * \brief The class that streams back validated utf-8 text strings
 * that generated by tokenizer.
 */
class TextStreamerObj : public Object {
 public:
  explicit TextStreamerObj(Tokenizer tokenizer);

  /*!
   * \brief Put new delta tokens into the streamer, and get the UTF-8-valid
   * delta string. The text streamer may hold some of the input delta tokens
   * which cannot decode into valid UTF-8 strings. The returned string
   * is always guaranteed to be UTF-8 valid.
   * \param delta_tokens The new tokens to put into the streamer.
   * \return The decoded delta string after putting the input new tokens.
   */
  std::string Put(const std::vector<int32_t>& delta_tokens);

  /*! \brief Return the string decoded by remaining tokens. */
  std::string Finish();

  // REPLACEMENT CHARACTER (U+FFFD) in UTF-8.
  static constexpr const char* kReplacementCharacter = "\xef\xbf\xbd";

  static constexpr const char* _type_key = "tapml.TextStreamer";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_BASE_OBJECT_INFO(TextStreamerObj, Object);

 private:
  Tokenizer tokenizer_;
  std::vector<int32_t> prefix_tokens_;
  std::vector<int32_t> pending_tokens_;
  bool finished_ = false;
};

/*!
 * \brief Managed reference to TextStreamerObj
 * \sa TextStreamerObj
 */
class TextStreamer : public ObjectRef {
 public:
  /*! \brief Construct a text streamer with tokenizer. */
  explicit TextStreamer(Tokenizer tokenizer);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TextStreamer, ObjectRef, TextStreamerObj);
};

/****************** StopStrHandler ******************/

/*!
 * \brief The stop string handler in TapML, which takes input delta tokens
 * one at a time, and return the output delta token before stopping due to
 * stop strings.
 */
class StopStrHandlerObj : public Object {
 public:
  explicit StopStrHandlerObj(Array<String> stop_strs, const std::vector<std::string>& token_table);

  /*!
   * \brief Add new input delta token to the handler, return output
   * delta tokens before stopping. The stop string handler may hold
   * some of the input delta token which may be part of a stop string.
   * The returned tokens are always guaranteed not to be part of stop string.
   */
  std::vector<int32_t> Put(int32_t token_id);

  /*! \brief Stop string handling has finished, return remaining cached token ids. */
  std::vector<int32_t> Finish() const { return pending_token_ids_; };

  /*! \brief Check if the generation has stopped due to stop string. */
  bool StopTriggered() const { return stop_triggered_; }

  static constexpr const char* _type_key = "tapml.StopStrHandler";
  TVM_DECLARE_FINAL_OBJECT_INFO(StopStrHandlerObj, Object);

 private:
  /*! \brief The stop strings. */
  Array<String> stop_strs_;
  /*! \brief The partial match table for each stop string in the KMP algorithm. */
  std::vector<std::vector<int>> partial_match_tables_;
  /*! \brief The tokenizer token table for token id lookup. */
  const std::vector<std::string>& token_table_;

  /************ Global states across all stop strings. ************/

  /*! \brief The globally pending string length. */
  int pending_string_len_ = 0;
  /*! \brief The globally pending token ids. */
  std::vector<int32_t> pending_token_ids_;
  /*! \brief The token string length of each pending token id. */
  std::vector<int> pending_token_lengths_;
  /*! \brief A boolean flag indicating if stop has been triggered. */
  bool stop_triggered_;

  /************ Per-stop-string states. ************/

  /*! \brief The current match position of the pending string to each stop string. */
  std::vector<int> cur_match_lengths_;
};

/*!
 * \brief Managed reference to StopStrHandlerObj
 * \sa StopStrHandlerObj
 */
class StopStrHandler : public ObjectRef {
 public:
  explicit StopStrHandler(Array<String> stop_strs, const std::vector<std::string>& token_table);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(StopStrHandler, ObjectRef, StopStrHandlerObj);
};

}  // namespace llm
}  // namespace tapml

#endif  // TAPML_STREAMER_H_
