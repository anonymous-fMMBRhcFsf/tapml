/*!
 * Copyright (c) 2023 by Contributors
 * \file progress_bar.h
 * \brief A simple progress bar in C++.
 */
#ifndef TapML_SUPPORT_PROGRESS_BAR_H_
#define TapML_SUPPORT_PROGRESS_BAR_H_

#include <iostream>
#include <string>

namespace tapml {
namespace llm {

class ProgressBar {
 public:
  explicit ProgressBar(int total, int width = 100) : total(total), width(width), cur(0) {}

  void Progress() {
    if (cur < total) {
      ++cur;
    }
    int bar_width = width - 2;  // Adjust for borders
    int completed = static_cast<int>(static_cast<float>(cur) / total * bar_width);
    int remaining = bar_width - completed;
    std::cout << "["                          //
              << std::string(completed, '=')  //
              << ">"                          //
              << std::string(remaining, ' ')  //
              << "] "                         //
              << " [" << cur << "/" << total << "]";
    if (cur < total) {
      std::cout << "\r";
      std::cout.flush();
    } else {
      std::cout << std::endl;  // Move to the next line after the progress bar is complete
    }
  }

 private:
  int total;
  int width;
  int cur;
};

}  // namespace llm
}  // namespace tapml

#endif  // TapML_SUPPORT_PROGRESS_BAR_H_
