// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#ifndef FPNGE_H
#define FPNGE_H
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct FPNGEOptions {
  char predictor; // 0-4: fixed predictor, 5: fast predictor selection, 6: slow
                  // predictor selection
};

#define FPNGE_COMPRESS_LEVEL_DEFAULT 4
#define FPNGE_COMPRESS_LEVEL_BEST 4
inline void FPNGEFillOptions(struct FPNGEOptions *options, int level) {
  if (level == 0)
    level = FPNGE_COMPRESS_LEVEL_DEFAULT;
  switch (level) {
  case 1:
    options->predictor = 2;
    break;
  case 2:
    options->predictor = 4;
    break;
  case 3:
    options->predictor = 5;
    break;
  default:
    options->predictor = 6;
    break;
  }
}

// bytes_per_channel = 1/2 for 8-bit and 16-bit. num_channels: 1/2/3/4
// (G/GA/RGB/RGBA)
size_t FPNGEEncode(size_t bytes_per_channel, size_t num_channels,
                   const void *data, size_t width, size_t row_stride,
                   size_t height, void *output,
                   const struct FPNGEOptions *options);

inline size_t FPNGEOutputAllocSize(size_t bytes_per_channel,
                                   size_t num_channels, size_t width,
                                   size_t height) {
  // likely an overestimate
  return 1024 + 2 * bytes_per_channel * num_channels * width * height;
}

#ifdef __cplusplus
}
#endif

#endif
