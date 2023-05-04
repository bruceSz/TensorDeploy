
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

#include "nv_utils.h"

#include <fstream>
#include <cassert>
#include <filesystem>
#include <NvInferRuntimeCommon.h>

#include "log.h"

namespace tensor {
namespace deploy {
void display_registered_plugins() {
  int p_nums = 0;
  const IPluginCreator* const plugin_creator =
      getPluginRegistry()->getPluginCreatorList(&p_nums);
  LOG(INFO) << "Registered plugins: " << p_nums;

  for (int i = 0; i < p_nums; ++i) {
    LOG(INFO) << "Plugin: " << plugin_creator[i].mPluginName;
  }
}

int get_engine_size(const std::string& engine_path) {
    assert(std::filesystem::exists(engine_path));
    std::ifstream engineFile(engineFilename, std::ios::binary);
    assert(engineFile.good());
    
    int fsize = 0;
    int ret = 0;
    // method 1.
    engineFile.seekg(0, std::ifstream::end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);
    ret = fsize;
    LOG(INFO) << "Engine file size: " << ret;

    // method 2.
    if (0) {
        std::vector<char> engineData(fsize);
        engineFile.read(engineData.data(), fsize);
        const char* buf = engineData.data();
        size_t size_trt = *(int *)(buf + 16) + 0x18;
        LOG(INFO) << "Inner engine size_trt: " << size_trt;
    }
    return ret;

}
}  // namespace deploy
}  // namespace tensor