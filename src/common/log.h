// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations



#pragma once

#include <cstdarg>
#include <iostream>
#include <string>
#include <string.h>

#include <glog/logging.h>



#define LEFT_BRACKET "["
#define RIGHT_BRACKET "] "

#ifndef MODULE_NAME
#define MODULE_NAME "tensorD"
#endif


// #if !defined(USECAST) // use glog api
// #define XDEBUG_MODULE(module) \
//   VLOG(4) << LEFT_BRACKET << module << RIGHT_BRACKET << "[DEBUG] "
// #define XDEBUG XDEBUG_MODULE(MODULE_NAME)
// #define XINFO XLOG_MODULE(MODULE_NAME, INFO)
// #define XWARN XLOG_MODULE(MODULE_NAME, WARN)
// #define XERROR XLOG_MODULE(MODULE_NAME, ERROR)
// #define XFATAL XLOG_MODULE(MODULE_NAME, FATAL)
#define XTRACE \
  std::cout << LEFT_BRACKET << module << RIGHT_BRACKET << "[TRACE] "

#ifndef XLOG_MODULE_STREAM
#define XLOG_MODULE_STREAM(log_severity) XLOG_MODULE_STREAM_##log_severity
#endif

#ifndef XLOG_MODULE
#define XLOG_MODULE(module, log_severity) \
  XLOG_MODULE_STREAM(log_severity)(module)
#endif

#define XLOG_MODULE_STREAM_INFO(module)                         \
  google::LogMessage(__FILE__, __LINE__, google::INFO).stream() \
      << LEFT_BRACKET << module << RIGHT_BRACKET

#define XLOG_MODULE_STREAM_WARN(module)                            \
  google::LogMessage(__FILE__, __LINE__, google::WARNING).stream() \
      << LEFT_BRACKET << module << RIGHT_BRACKET

#define XLOG_MODULE_STREAM_ERROR(module)                         \
  google::LogMessage(__FILE__, __LINE__, google::ERROR).stream() \
      << LEFT_BRACKET << module << RIGHT_BRACKET

#define XLOG_MODULE_STREAM_FATAL(module)                         \
  google::LogMessage(__FILE__, __LINE__, google::FATAL).stream() \
      << LEFT_BRACKET << module << RIGHT_BRACKET

#define XINFO_IF(cond) XLOG_IF(INFO, cond, MODULE_NAME)
#define XWARN_IF(cond) XLOG_IF(WARN, cond, MODULE_NAME)
#define XERROR_IF(cond) XLOG_IF(ERROR, cond, MODULE_NAME)
#define XFATAL_IF(cond) XLOG_IF(FATAL, cond, MODULE_NAME)

#define XDEBUG_IF(cond) XLOG_IF2(DEBUG, cond)
#define XLOG_IF2(severity, cond) \
  !(cond) ? (void)0 : X##severity

#define XLOG_IF(severity, cond, module) \
  !(cond) ? (void)0                     \
          : google::LogMessageVoidify() & XLOG_MODULE(module, severity)

#define XCHECK(cond) CHECK(cond) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET
#define XCHECK_GE(value1, value2) XCHECK(value1 >= value2)
#define XCHECK_GT(value1, value2) XCHECK(value1 > value2)
#define XCHECK_EQ(value1, value2) XCHECK(value1 == value2)
#define XCHECK_LE(value1, value2) XCHECK(value1 <= value2)
#define XCHECK_LT(value1, value2) XCHECK(value1 < value2)
#define XCHECK_NE(value1, value2) XCHECK(value1 != value2)
#define XCHECK_NOTNULL(value) XCHECK(value != nullptr)

#define __SAFE_XCHECK(cond, info)         \
  if (cond == false) {                    \
    std::stringstream ss;                 \
    ss << info;                           \
    throw std::runtime_error(ss.str());   \
  }                                       \

#define SAFE_XCHECK(cond, info) __SAFE_XCHECK(cond, info)
#define SAFE_XCHECK_GE(value1, value2, info) SAFE_XCHECK(value1 >= value2, info)
#define SAFE_XCHECK_GT(value1, value2, info) SAFE_XCHECK(value1 > value2, info)
#define SAFE_XCHECK_EQ(value1, value2, info) SAFE_XCHECK(value1 == value2, info)
#define SAFE_XCHECK_LE(value1, value2, info) SAFE_XCHECK(value1 <= value2, info)
#define SAFE_XCHECK_LT(value1, value2, info) SAFE_XCHECK(value1 < value2, info)
#define SAFE_XCHECK_NE(value1, value2, info) SAFE_XCHECK(value1 != value2, info)

#define XINFO_EVERY(freq) \
  LOG_EVERY_N(INFO, freq) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET
#define XWARN_EVERY(freq) \
  LOG_EVERY_N(WARNING, freq) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET
#define XERROR_EVERY(freq) \
  LOG_EVERY_N(ERROR, freq) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET
#define XDEBUG_EVERY(freq) \
  VLOG_EVERY_N(4, freq) << LEFT_BRACKET << MODULE_NAME << RIGHT_BRACKET << "[DEBUG] "





#if !defined(RETURN_IF_NULL)
#define RETURN_IF_NULL(ptr)          \
  if (ptr == nullptr) {              \
    XWARN << #ptr << " is nullptr."; \
    return;                          \
  }
#endif

#if !defined(RETURN_VAL_IF_NULL)
#define RETURN_VAL_IF_NULL(ptr, val) \
  if (ptr == nullptr) {              \
    XWARN << #ptr << " is nullptr."; \
    return val;                      \
  }
#endif

#if !defined(RETURN_IF)
#define RETURN_IF(condition)           \
  if (condition) {                     \
    XWARN << #condition << " is met."; \
    return;                            \
  }
#endif

#if !defined(RETURN_VAL_IF)
#define RETURN_VAL_IF(condition, val)  \
  if (condition) {                     \
    XWARN << #condition << " is met."; \
    return val;                        \
  }
#endif

#if !defined(_RETURN_VAL_IF_NULL2__)
#define _RETURN_VAL_IF_NULL2__
#define RETURN_VAL_IF_NULL2(ptr, val) \
  if (ptr == nullptr) {               \
    return (val);                     \
  }
#endif

#if !defined(_RETURN_VAL_IF2__)
#define _RETURN_VAL_IF2__
#define RETURN_VAL_IF2(condition, val) \
  if (condition) {                     \
    return (val);                      \
  }
#endif

#if !defined(_RETURN_IF2__)
#define _RETURN_IF2__
#define RETURN_IF2(condition) \
  if (condition) {            \
    return;                   \
  }
#endif

#if !defined(_CONTINUE_IF__)
#define _CONTINUE_IF__
#define CONTINUE_IF(condition)             \
  if (condition) {                         \
    XWARN << #condition << " is not met."; \
    continue;                              \
  }
#endif

#if !defined(_CONTINUE_IF2__)
#define _CONTINUE_IF2__
#define CONTINUE_IF2(condition) \
  if (condition) {              \
    continue;                   \
  }
#endif

namespace tensor {
namespace deploy {
// #if !defined(USECAST)
void GoogleInitLogger(const std::string &bin_name = std::string(""),
                const std::string &level = std::string(""),
                const std::string &log_dir = std::string(""),
                const std::string &async_log = std::string(""));
bool IsLoggerInited();

void MarkLoggerInited(bool flag = true);
// #else
void InitLogger(const std::string &bin_name = std::string(""),
                const std::string &level = std::string(""),
                const std::string &log_dir = std::string(""),
                const std::string &async_log = std::string(""),
                const std::string &type = std::string("TEXT"));
// #endif


}  // namespace deploy
}  // namespace tensor
