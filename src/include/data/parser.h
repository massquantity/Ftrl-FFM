#ifndef FTRL_FFM_PARSER_H
#define FTRL_FFM_PARSER_H

#include <regex>
#include <string>

#include "data/sample.h"

namespace ftrl {

class Parser {
 public:
  Parser() = default;
  virtual ~Parser() = default;
  virtual void parse(const std::string &line, Sample &sample) = 0;
};

class LibsvmParser : public Parser {
 public:
  void parse(const std::string &line, Sample &sample) override;
};

class FFMParser : public Parser {
 public:
  void parse(const std::string &line, Sample &sample) override;
  [[maybe_unused]] void parseFFM(const std::string &line, Sample &sample);
  [[maybe_unused]] void parseCFFM(const std::string &line, Sample &sample);

 private:
  std::regex reg{"([[:digit:]]+):([[:digit:]]+):([[:digit:]]+\\.?[[:digit:]]*)"};
};

}  // namespace ftrl

#endif  // FTRL_FFM_PARSER_H
