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

#include <algorithm>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>


#if defined(_MSC_VER) && !defined(__clang__)
# define FORCE_INLINE_LAMBDA [[msvc::forceinline]]
# define FORCE_INLINE __forceinline
# define __SSE4_1__ 1
# ifdef __AVX2__
#  define __BMI2__ 1
# endif
#else
# define FORCE_INLINE_LAMBDA __attribute__((always_inline))
# define FORCE_INLINE __attribute__((always_inline)) inline
#endif

#if defined(__x86_64__) || \
  defined(__amd64__ ) || \
  defined(__LP64    ) || \
  defined(_M_X64    ) || \
  defined(_M_AMD64  ) || \
  (defined(_WIN64) && !defined(_M_ARM64))
# define PLATFORM_AMD64 1
#endif

#if defined(__tune_atom__) || defined(__tune_slm__) || defined(__tune_btver1__)
# define FPNGE_SLOW_PSHUFB
#endif

#include <wmmintrin.h>  // for CLMUL

#ifdef __AVX2__
# include <immintrin.h>
# define _mm(f) _mm256_##f
# define _mmsi(f) _mm256_##f##_si256
# define __mivec __m256i
# define BCAST128 _mm256_broadcastsi128_si256
# define INT2VEC(v) _mm256_castsi128_si256(_mm_cvtsi32_si128(v))
# define SIMD_WIDTH 32
# define SIMD_MASK 0xffffffffU
#elif defined(__SSE4_1__)
# include <nmmintrin.h>
# define _mm(f) _mm_##f
# define _mmsi(f) _mm_##f##_si128
# define __mivec __m128i
# define BCAST128(v) (v)
# define INT2VEC _mm_cvtsi32_si128
# define SIMD_WIDTH 16
# define SIMD_MASK 0xffffU
#else
# error Requires SSE4.1 support minium
#endif


alignas(16) constexpr uint8_t kBitReverseNibbleLookup[16] = {
    0b0000, 0b1000, 0b0100, 0b1100, 0b0010, 0b1010, 0b0110, 0b1110,
    0b0001, 0b1001, 0b0101, 0b1101, 0b0011, 0b1011, 0b0111, 0b1111,
};

static constexpr uint8_t kLZ77NBits[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                           1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                           4, 4, 4, 4, 5, 5, 5, 5, 0};

static constexpr uint16_t kLZ77Base[29] = {
    3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
    31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};

static uint16_t BitReverse(size_t nbits, uint16_t bits) {
  uint16_t rev16 = (kBitReverseNibbleLookup[bits & 0xF] << 12) |
                   (kBitReverseNibbleLookup[(bits >> 4) & 0xF] << 8) |
                   (kBitReverseNibbleLookup[(bits >> 8) & 0xF] << 4) |
                   (kBitReverseNibbleLookup[bits >> 12]);
  return rev16 >> (16 - nbits);
}

struct HuffmanTable {
  uint8_t nbits[286];
  uint16_t end_bits;

  alignas(16) uint8_t first16_nbits[16];
  alignas(16) uint8_t first16_bits[16];

  alignas(16) uint8_t last16_nbits[16];
  alignas(16) uint8_t last16_bits[16];

  alignas(16) uint8_t mid_lowbits[16];
  uint8_t mid_nbits;

  uint32_t lz77_length_nbits[259] = {};
  uint32_t lz77_length_bits[259] = {};
  uint32_t lz77_length_sym[259] = {};

  // Computes nbits[i] for i <= n, subject to min_limit[i] <= nbits[i] <=
  // max_limit[i], so to minimize sum(nbits[i] * freqs[i]).
  static void ComputeCodeLengths(const uint64_t *freqs, size_t n, uint8_t *min_limit,
                                 uint8_t *max_limit, uint8_t *nbits) {
    size_t precision = 0;
    uint64_t freqsum = 0;
    for (size_t i = 0; i < n; i++) {
      assert(freqs[i] != 0);
      freqsum += freqs[i];
      if (min_limit[i] < 1)
        min_limit[i] = 1;
      assert(min_limit[i] <= max_limit[i]);
      precision = std::max<size_t>(max_limit[i], precision);
    }
    uint64_t infty = freqsum * precision;
    std::vector<uint64_t> dynp(((1U << precision) + 1) * (n + 1), infty);
    auto d = [&](size_t sym, size_t off) -> uint64_t & {
      return dynp[sym * ((1 << precision) + 1) + off];
    };
    d(0, 0) = 0;
    for (size_t sym = 0; sym < n; sym++) {
      for (size_t bits = min_limit[sym]; bits <= max_limit[sym]; bits++) {
        size_t off_delta = 1U << (precision - bits);
        for (size_t off = 0; off + off_delta <= (1U << precision); off++) {
          d(sym + 1, off + off_delta) = std::min(
              d(sym, off) + freqs[sym] * bits, d(sym + 1, off + off_delta));
        }
      }
    }

    size_t sym = n;
    size_t off = 1U << precision;

    while (sym-- > 0) {
      assert(off > 0);
      for (size_t bits = min_limit[sym]; bits <= max_limit[sym]; bits++) {
        size_t off_delta = 1U << (precision - bits);
        if (off_delta <= off &&
            d(sym + 1, off) == d(sym, off - off_delta) + freqs[sym] * bits) {
          off -= off_delta;
          nbits[sym] = bits;
          break;
        }
      }
    }
  }

  void ComputeNBits(const uint64_t *collected_data) {
    constexpr uint64_t kBaselineData[286] = {
        113, 54, 28, 18, 12, 9, 7, 6, 5, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 7, 9,
        12,  18, 29, 54, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1,   1,  1,  1,  1,  1, 1, 1, 1, 1, 1, 1, 1,
    };

    uint64_t data[286];

    for (size_t i = 0; i < 286; i++) {
      data[i] = collected_data[i] + kBaselineData[i];
    }

    // Compute Huffman code length ensuring that all the "fake" symbols for [16,
    // 240) and [255, 285) have their maximum length.
    uint64_t collapsed_data[16 + 14 + 16 + 2] = {};
    uint8_t collapsed_min_limit[16 + 14 + 16 + 2] = {};
    uint8_t collapsed_max_limit[16 + 14 + 16 + 2];
    for (size_t i = 0; i < 48; i++) {
      collapsed_max_limit[i] = 8;
    }
    for (size_t i = 0; i < 16; i++) {
      collapsed_data[i] = data[i];
    }
    for (size_t i = 0; i < 14; i++) {
      collapsed_data[16 + i] = 1;
      collapsed_min_limit[16 + i] = 8 * 0;
    }
    for (size_t j = 0; j < 16; j++) {
      collapsed_data[16 + 14 + j] += data[240 + j];
    }
    collapsed_data[16 + 14 + 16] = 1;
    collapsed_min_limit[16 + 14 + 16] = 8 * 0;
    collapsed_data[16 + 14 + 16 + 1] = data[285];

    uint8_t collapsed_nbits[48] = {};
    ComputeCodeLengths(collapsed_data, 48, collapsed_min_limit,
                       collapsed_max_limit, collapsed_nbits);

    // Compute "extra" code lengths for symbols >= 256, except 285.
    uint8_t tail_nbits[29] = {};
    uint8_t tail_min_limit[29] = {};
    uint8_t tail_max_limit[29] = {};
    for (size_t i = 0; i < 29; i++) {
      tail_min_limit[i] = 4;
      tail_max_limit[i] = 7;
    }
    ComputeCodeLengths(data + 256, 29, tail_min_limit, tail_max_limit,
                       tail_nbits);

    for (size_t i = 0; i < 16; i++) {
      nbits[i] = collapsed_nbits[i];
    }
    for (size_t i = 0; i < 14; i++) {
      for (size_t j = 0; j < 16; j++) {
        nbits[(i + 1) * 16 + j] = collapsed_nbits[16 + i] + 4;
      }
    }
    for (size_t i = 0; i < 16; i++) {
      nbits[240 + i] = collapsed_nbits[30 + i];
    }
    for (size_t i = 0; i < 29; i++) {
      nbits[256 + i] = collapsed_nbits[46] + tail_nbits[i];
    }
    nbits[285] = collapsed_nbits[47];
  }

  void ComputeCanonicalCode(const uint8_t *nbits, uint16_t *bits) {
    uint8_t code_length_counts[16] = {};
    for (size_t i = 0; i < 286; i++) {
      code_length_counts[nbits[i]]++;
    }
    uint16_t next_code[16] = {};
    uint16_t code = 0;
    for (size_t i = 1; i < 16; i++) {
      code = (code + code_length_counts[i - 1]) << 1;
      next_code[i] = code;
    }
    for (size_t i = 0; i < 286; i++) {
      bits[i] = BitReverse(nbits[i], next_code[nbits[i]]++);
    }
  }

  HuffmanTable(const uint64_t *collected_data) {
    ComputeNBits(collected_data);
    uint16_t bits[286];
    ComputeCanonicalCode(nbits, bits);
    for (size_t i = 0; i < 16; i++) {
      first16_nbits[i] = nbits[i];
      first16_bits[i] = bits[i];
    }
    for (size_t i = 0; i < 16; i++) {
      last16_nbits[i] = nbits[240 + i];
      last16_bits[i] = bits[240 + i];
    }
    mid_nbits = nbits[16];
    mid_lowbits[0] = mid_lowbits[15] = 0;
    for (size_t i = 16; i < 240; i += 16) {
      mid_lowbits[i / 16] = bits[i] & ((1U << (mid_nbits - 4)) - 1);
    }
    for (size_t i = 16; i < 240; i++) {
      assert(nbits[i] == mid_nbits);
      assert((uint32_t(mid_lowbits[i / 16]) |
              (kBitReverseNibbleLookup[i % 16] << (mid_nbits - 4))) == bits[i]);
    }
    end_bits = bits[256];
    // Construct lz77 lookup tables.
    for (size_t i = 0; i < 29; i++) {
      for (size_t j = 0; j < (1U << kLZ77NBits[i]); j++) {
        lz77_length_nbits[kLZ77Base[i] + j] = nbits[257 + i] + kLZ77NBits[i];
        lz77_length_sym[kLZ77Base[i] + j] = 257 + i;
        lz77_length_bits[kLZ77Base[i] + j] =
            bits[257 + i] | (j << nbits[257 + i]);
      }
    }
  }
};

struct BitWriter {
  void Write(uint32_t count, uint64_t bits) {
    buffer |= bits << bits_in_buffer;
    bits_in_buffer += count;
    memcpy(data + bytes_written, &buffer, 8);
    size_t bytes_in_buffer = bits_in_buffer / 8;
    bits_in_buffer &= 7;
    buffer >>= bytes_in_buffer * 8;
    bytes_written += bytes_in_buffer;
  }

  void ZeroPadToByte() {
    if (bits_in_buffer != 0) {
      Write(8 - bits_in_buffer, 0);
    }
  }

  unsigned char *data;
  size_t bytes_written = 0;
  size_t bits_in_buffer = 0;
  uint64_t buffer = 0;
};

static void WriteHuffmanCode(uint32_t &dist_nbits, uint32_t &dist_bits,
                      const HuffmanTable &table, BitWriter *__restrict writer) {
  dist_nbits = 1;
  dist_bits = 0;

  constexpr uint8_t kCodeLengthNbits[] = {
      4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0,
  };
  constexpr uint8_t kCodeLengthOrder[] = {
      16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
  };
  writer->Write(5, 29); // all lit/len codes
  writer->Write(5, 0);  // distance code up to dist, included
  writer->Write(4, 15); // all code length codes
  for (size_t i = 0; i < 19; i++) {
    writer->Write(3, kCodeLengthNbits[kCodeLengthOrder[i]]);
  }

  for (size_t i = 0; i < 286; i++) {
    writer->Write(4, kBitReverseNibbleLookup[table.nbits[i]]);
  }
  writer->Write(4, 0b1000);
}

constexpr unsigned kCrcTable[] = {
    0x0,        0x77073096, 0xee0e612c, 0x990951ba, 0x76dc419,  0x706af48f,
    0xe963a535, 0x9e6495a3, 0xedb8832,  0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
    0x9b64c2b,  0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
    0xf3b97148, 0x84be41de, 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
    0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec, 0x14015c4f, 0x63066cd9,
    0xfa0f3d63, 0x8d080df5, 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
    0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b, 0x35b5a8fa, 0x42b2986c,
    0xdbbbc9d6, 0xacbcf940, 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
    0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
    0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
    0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d, 0x76dc4190, 0x1db7106,
    0x98d220bc, 0xefd5102a, 0x71b18589, 0x6b6b51f,  0x9fbfe4a5, 0xe8b8d433,
    0x7807c9a2, 0xf00f934,  0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x86d3d2d,
    0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
    0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
    0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
    0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
    0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
    0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
    0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
    0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
    0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x3b6e20c,  0x74b1d29a,
    0xead54739, 0x9dd277af, 0x4db2615,  0x73dc1683, 0xe3630b12, 0x94643b84,
    0xd6d6a3e,  0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0xa00ae27,  0x7d079eb1,
    0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
    0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
    0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
    0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
    0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
    0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
    0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
    0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
    0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x26d930a,  0x9c0906a9, 0xeb0e363f,
    0x72076785, 0x5005713,  0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0xcb61b38,
    0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0xbdbdf21,  0x86d3d2d4, 0xf1d4e242,
    0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
    0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
    0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
    0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
    0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
    0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
    0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
    0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d};

static unsigned long update_crc(unsigned long crc, const unsigned char *buf, int len) {
  static const uint64_t k1k2[] = {0x1'5444'2BD4ULL, 0x1'C6E4'1596ULL};
  static const uint64_t k3k4[] = {0x1'7519'97D0ULL, 0x0'CCAA'009EULL};
  static const uint64_t k5k6[] = {0x1'63CD'6124ULL, 0x0'0000'0000ULL};
  static const uint64_t poly[] = {0x1'DB71'0641ULL, 0x1'F701'1641ULL};

  int n = 0;
  unsigned long c;

  if (len >= 128) {
    // Adapted from WUFFs code.
    auto x0 = _mm_loadu_si128((__m128i *)(buf + n + 0x00));
    auto x1 = _mm_loadu_si128((__m128i *)(buf + n + 0x10));
    auto x2 = _mm_loadu_si128((__m128i *)(buf + n + 0x20));
    auto x3 = _mm_loadu_si128((__m128i *)(buf + n + 0x30));

    x0 = _mm_xor_si128(x0, _mm_cvtsi32_si128(crc));
    n += 64;

    auto k = _mm_loadu_si128((__m128i *)k1k2);
    while (n + 64 <= len) {
      auto y0 = _mm_clmulepi64_si128(x0, k, 0x00);
      auto y1 = _mm_clmulepi64_si128(x1, k, 0x00);
      auto y2 = _mm_clmulepi64_si128(x2, k, 0x00);
      auto y3 = _mm_clmulepi64_si128(x3, k, 0x00);

      x0 = _mm_clmulepi64_si128(x0, k, 0x11);
      x1 = _mm_clmulepi64_si128(x1, k, 0x11);
      x2 = _mm_clmulepi64_si128(x2, k, 0x11);
      x3 = _mm_clmulepi64_si128(x3, k, 0x11);

      x0 = _mm_xor_si128(_mm_xor_si128(x0, y0),
                         _mm_loadu_si128((__m128i *)(buf + n + 0x00)));
      x1 = _mm_xor_si128(_mm_xor_si128(x1, y1),
                         _mm_loadu_si128((__m128i *)(buf + n + 0x10)));
      x2 = _mm_xor_si128(_mm_xor_si128(x2, y2),
                         _mm_loadu_si128((__m128i *)(buf + n + 0x20)));
      x3 = _mm_xor_si128(_mm_xor_si128(x3, y3),
                         _mm_loadu_si128((__m128i *)(buf + n + 0x30)));
      n += 64;
    }

    k = _mm_loadu_si128((__m128i *)k3k4);
    auto y0 = _mm_clmulepi64_si128(x0, k, 0x00);
    x0 = _mm_clmulepi64_si128(x0, k, 0x11);
    x0 = _mm_xor_si128(x0, x1);
    x0 = _mm_xor_si128(x0, y0);
    y0 = _mm_clmulepi64_si128(x0, k, 0x00);
    x0 = _mm_clmulepi64_si128(x0, k, 0x11);
    x0 = _mm_xor_si128(x0, x2);
    x0 = _mm_xor_si128(x0, y0);
    y0 = _mm_clmulepi64_si128(x0, k, 0x00);
    x0 = _mm_clmulepi64_si128(x0, k, 0x11);
    x0 = _mm_xor_si128(x0, x3);
    x0 = _mm_xor_si128(x0, y0);

    x1 = _mm_clmulepi64_si128(x0, k, 0x10);
    x2 = _mm_setr_epi32(~0U, 0, ~0U, 0);
    x0 = _mm_srli_si128(x0, 8);
    x0 = _mm_xor_si128(x0, x1);

    k = _mm_loadu_si128((__m128i *)k5k6);
    x1 = _mm_srli_si128(x0, 4);
    x0 = _mm_and_si128(x0, x2);
    x0 = _mm_clmulepi64_si128(x0, k, 0x00);
    x0 = _mm_xor_si128(x0, x1);

    k = _mm_loadu_si128((__m128i *)poly);
    x1 = _mm_and_si128(x0, x2);
    x1 = _mm_clmulepi64_si128(x1, k, 0x10);
    x1 = _mm_and_si128(x1, x2);
    x1 = _mm_clmulepi64_si128(x1, k, 0x00);
    x0 = _mm_xor_si128(x0, x1);

    c = _mm_extract_epi32(x0, 1);
  } else {
    c = crc;
  }

  for (; n < len; n++) {
    c = kCrcTable[(c ^ buf[n]) & 0xff] ^ (c >> 8);
  }
  return c;
}

static unsigned long compute_crc(const unsigned char *buf, int len) {
  return update_crc(0xffffffffL, buf, len) ^ 0xffffffffL;
}

constexpr unsigned kAdler32Mod = 65521;

static void UpdateAdler32(uint32_t &s1, uint32_t &s2, uint8_t byte) {
  s1 += byte;
  s2 += s1;
  s1 %= kAdler32Mod;
  s2 %= kAdler32Mod;
}

static uint32_t hadd(__mivec v) {
  auto sum =
#ifdef __AVX2__
      _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
#else
      v;
#endif
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

template <size_t predictor>
static FORCE_INLINE __mivec
PredictVec(const unsigned char *current_buf, const unsigned char *top_buf,
           const unsigned char *left_buf, const unsigned char *topleft_buf) {
  auto data = _mmsi(load)((__mivec *)(current_buf));
  if (predictor == 0) {
    return data;
  } else if (predictor == 1) {
    auto pred = _mmsi(loadu)((__mivec *)(left_buf));
    return _mm(sub_epi8)(data, pred);
  } else if (predictor == 2) {
    auto pred = _mmsi(load)((__mivec *)(top_buf));
    return _mm(sub_epi8)(data, pred);
  } else if (predictor == 3) {
    auto left = _mmsi(loadu)((__mivec *)(left_buf));
    auto top = _mmsi(load)((__mivec *)(top_buf));
    auto pred = _mm(avg_epu8)(top, left);
    // emulate truncating average
    pred = _mm(sub_epi8)(pred, _mmsi(and)(_mmsi(xor)(top, left),
                                                  _mm(set1_epi8)(1)));
    return _mm(sub_epi8)(data, pred);
  } else {
    auto a = _mmsi(loadu)((__mivec *)(left_buf));
    auto b = _mmsi(load)((__mivec *)(top_buf));
    auto c = _mmsi(loadu)((__mivec *)(topleft_buf));
    auto min_bc = _mm(min_epu8)(b, c);
    auto min_ac = _mm(min_epu8)(a, c);
    auto pa = _mm(sub_epi8)(_mm(max_epu8)(b, c), min_bc);
    auto pb = _mm(sub_epi8)(_mm(max_epu8)(a, c), min_ac);
    auto min_pab = _mm(min_epu8)(pa, pb);
    auto pc = _mm(sub_epi8)(_mm(max_epu8)(pa, pb), min_pab);
    pc = _mmsi(or)(pc, _mmsi(xor)(
      _mm(cmpeq_epi8)(min_bc, c),
      _mm(cmpeq_epi8)(min_ac, a)
    ));
    
    auto use_a = _mm(cmpeq_epi8)(_mm(min_epu8)(min_pab, pc), pa);
    auto use_b = _mm(cmpeq_epi8)(_mm(min_epu8)(pb, pc), pb);

    auto pred = _mm(blendv_epi8)(_mm(blendv_epi8)(c, b, use_b), a, use_a);
    return _mm(sub_epi8)(data, pred);
    /*
    // Equivalent scalar code:
    for (size_t ii = 0; ii < 32; ii++) {
      uint8_t a = left_buf[i + ii];
      uint8_t b = top_buf[i + ii];
      uint8_t c = topleft_buf[i + ii];
      uint8_t bc = b - c;
      uint8_t ca = c - a;
      uint8_t pa = c < b ? bc : -bc;
      uint8_t pb = a < c ? ca : -ca;
      uint8_t pc = (a < c) == (c < b) ? (bc >= ca ? bc - ca : ca - bc) : 255;
      uint8_t pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      uint8_t data = current_row_buf[i + ii] - pred;
      predicted_data[ii] = data;
    }
    */
  }
}

alignas(SIMD_WIDTH) constexpr int32_t _kMaskVec[] = {
    -1, -1, -1, -1,
#if SIMD_WIDTH == 32
    -1, -1, -1, -1,
    0, 0, 0, 0,
#endif
    0, 0, 0, 0
};
static const uint8_t* kMaskVec = (const uint8_t *)_kMaskVec + SIMD_WIDTH;

template <size_t predictor, typename CB, typename CB_ADL, typename CB_RLE>
static void ProcessRow(size_t bytes_per_line,
           const unsigned char *current_row_buf, const unsigned char *top_buf,
           const unsigned char *left_buf, const unsigned char *topleft_buf,
           CB &&cb, CB_ADL &&cb_adl, CB_RLE &&cb_rle) {
  size_t run = 0;
  size_t i = 0;
  for (; i + SIMD_WIDTH <= bytes_per_line; i += SIMD_WIDTH) {
    auto pdata = PredictVec<predictor>(current_row_buf + i, top_buf + i,
                                      left_buf + i, topleft_buf + i);
    unsigned pdatais0 = _mm(movemask_epi8)(_mm(cmpeq_epi8)(pdata, _mmsi(setzero)()));

    if (pdatais0 == SIMD_MASK) {
      run += SIMD_WIDTH;
    } else {
      if (run != 0) {
        cb_rle(run);
      }
      run = 0;
      cb(pdata, SIMD_WIDTH);
    }
    cb_adl(pdata, SIMD_WIDTH);
  }
  
  size_t bytes_remaining = bytes_per_line ^ i;
  if (bytes_remaining) {
    auto pdata = PredictVec<predictor>(current_row_buf + i, top_buf + i,
                                      left_buf + i, topleft_buf + i);
    auto pdatais0 = _mm(movemask_epi8)(_mm(cmpeq_epi8)(pdata, _mmsi(setzero)()));
    auto mask = (1UL << bytes_remaining) - 1;

    if ((pdatais0 & mask) == mask && run + bytes_remaining >= 16) {
      run += bytes_remaining;
    } else {
      if (run != 0) {
        cb_rle(run);
      }
      run = 0;
      cb(pdata, bytes_remaining);
    }
    cb_adl(pdata, bytes_remaining);
  }
  if (run != 0) {
    cb_rle(run);
  }
}

template <typename CB> static void ForAllRLESymbols(size_t length, CB &&cb) {
  assert(length >= 4);
  length -= 1;

  if (length < 258) {
    // fast path if long sequences are rare in the image
    cb(length, 1);
  } else {
    auto runs = length / 258;
    auto remain = length % 258;
    if (remain == 1 || remain == 2) {
      remain += 258 - 3;
      runs--;
      cb(3, 1);
    }
    if (runs) {
      cb(258, runs);
    }
    if (remain) {
      cb(remain, 1);
    }
  }
}

template <size_t pred>
static void TryPredictor(size_t bytes_per_line,
                  const unsigned char *current_row_buf, const unsigned char *top_buf,
                  const unsigned char *left_buf,
                  const unsigned char *topleft_buf, const HuffmanTable &table,
                  size_t &best_cost, uint8_t &predictor, size_t dist_nbits) {
  size_t cost_rle = 0;
  __mivec cost_direct = _mmsi(setzero)();
  auto cost_chunk_cb = [&](const __mivec bytes, const size_t bytes_in_vec)
      FORCE_INLINE_LAMBDA {

    auto data_for_lut = _mmsi(and)(_mm(set1_epi8)(0xF), bytes);
    auto use_lowhi = _mm(cmpgt_epi8)(
      _mm(add_epi8)(bytes, _mm(set1_epi8)(112)),
      _mm(set1_epi8)(95)
    );

    auto nbits_low16 =
        _mm(shuffle_epi8)(BCAST128(
                                _mm_load_si128((__m128i *)table.first16_nbits)),
                            data_for_lut);
    auto nbits_hi16 =
        _mm(shuffle_epi8)(BCAST128(
                                _mm_load_si128((__m128i *)table.last16_nbits)),
                            data_for_lut);

    auto nbits = _mm(blendv_epi8)(
      _mm(set1_epi8)(table.mid_nbits),
      _mm(blendv_epi8)(nbits_low16, nbits_hi16, bytes),
      use_lowhi
    );

    nbits = _mmsi(and)(nbits, _mmsi(loadu)((__mivec *)(kMaskVec - bytes_in_vec)));

    cost_direct = _mm(add_epi32)(
        cost_direct, _mm(sad_epu8)(nbits, _mmsi(setzero)()));
  };
  ProcessRow<pred>(
      bytes_per_line, current_row_buf, top_buf, left_buf, topleft_buf,
      cost_chunk_cb, [](const __mivec, const size_t) {},
      [&](size_t run) {
        cost_rle += table.first16_nbits[0];
        ForAllRLESymbols(run, [&](size_t len, size_t count) {
          cost_rle += (dist_nbits + table.lz77_length_nbits[len]) * count;
        });
      });
  size_t cost = cost_rle + hadd(cost_direct);
  if (cost < best_cost) {
    best_cost = cost;
    predictor = pred;
  }
}

static FORCE_INLINE void WriteBits(__mivec nbits, __mivec bits_lo,
                                              __mivec bits_hi,
                                              size_t mid_lo_nbits,
                                              BitWriter *__restrict writer) {

#if defined(__BMI2__) && defined(PLATFORM_AMD64) && \
  !defined(__tune_bdver4__) && !defined(__tune_znver1__) && !defined(__tune_znver2__)
# define USE_PEXT 1
#endif
#ifdef __AVX2__
  constexpr uint8_t kPerm[] = {0, 1, 4, 5, 2, 3, 6, 7};
#else
  constexpr uint8_t kPerm[] = {0, 1, 2, 3};
#endif

  // Merge bits_lo and bits_hi in 16-bit "bits".
#ifdef USE_PEXT
  auto bits0 = _mm(unpacklo_epi8)(bits_lo, bits_hi);
  auto bits1 = _mm(unpackhi_epi8)(bits_lo, bits_hi);
  
  // convert nbits into a mask
  auto nbits_hi = _mm(sub_epi8)(nbits, _mm(set1_epi8)(mid_lo_nbits));
  auto nbits0 = _mm(unpacklo_epi8)(nbits, nbits_hi);
  auto nbits1 = _mm(unpackhi_epi8)(nbits, nbits_hi);
  const auto nbits_to_mask = BCAST128(_mm_set_epi32(
    0xffffffff, 0xffffffff, 0x7f3f1f0f, 0x07030100
  ));
  auto bitmask0 = _mm(shuffle_epi8)(nbits_to_mask, nbits0);
  auto bitmask1 = _mm(shuffle_epi8)(nbits_to_mask, nbits1);
  
  // aggregate nbits
  alignas(16) uint16_t nbits_a[SIMD_WIDTH/4];
  auto bc = _mm(maddubs_epi16)(nbits, _mm(set1_epi8)(1));
  __m128i bit_count;
# ifdef __AVX2__
  bit_count = _mm_hadd_epi16(_mm256_castsi256_si128(bc), _mm256_extracti128_si256(bc, 1));
# else
  bit_count = _mm(hadd_epi16)(bc, bc);
# endif
  
  alignas(SIMD_WIDTH) uint64_t bits_a[SIMD_WIDTH/4];
  _mmsi(store)((__mivec *)bits_a, bits0);
  _mmsi(store)((__mivec *)bits_a + 1, bits1);
  alignas(SIMD_WIDTH) uint64_t bitmask_a[SIMD_WIDTH/4];
  _mmsi(store)((__mivec *)bitmask_a, bitmask0);
  _mmsi(store)((__mivec *)bitmask_a + 1, bitmask1);
  
# ifdef __AVX2__
  bit_count = _mm_shuffle_epi32(bit_count, _MM_SHUFFLE(3, 1, 2, 0));
  _mm_store_si128((__m128i *)nbits_a, bit_count);
# else
  _mm_storel_epi64((__m128i *)nbits_a, bit_count);
# endif
  
#else
# ifdef FPNGE_SLOW_PSHUFB
  auto nbits_mix = _mm(unpacklo_epi8)(nbits, _mm(unpackhi_epi64)(nbits, nbits));
# else
  auto nbits_mix = _mm(shuffle_epi8)(nbits, BCAST128(_mm_set_epi32(
    0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800
  )));
# endif
  __mivec bits0, bits1;
  if (mid_lo_nbits == 8) {
    bits0 = _mm(unpacklo_epi8)(bits_lo, bits_hi);
    bits1 = _mm(unpackhi_epi8)(bits_lo, bits_hi);
  } else {
    auto nbits_shift = _mm_cvtsi32_si128(8 - mid_lo_nbits);
    auto bits_lo_shifted = _mm(sll_epi16)(bits_lo, nbits_shift);
    bits0 = _mm(unpacklo_epi8)(bits_lo_shifted, bits_hi);
    bits1 = _mm(unpackhi_epi8)(bits_lo_shifted, bits_hi);
    
    bits0 = _mm(srl_epi16)(bits0, nbits_shift);
    bits1 = _mm(srl_epi16)(bits1, nbits_shift);
  }
  
  // 16 -> 32
  auto nbits0_32_lo = _mmsi(and)(nbits_mix, _mm(set1_epi32)(0xff));
  auto nbits1_32_lo = _mmsi(and)(_mm(srli_epi16)(nbits_mix, 8), _mm(set1_epi32)(0xff));
  auto bits0_32_lo = _mmsi(and)(bits0, _mm(set1_epi32)(0xFFFF));
  auto bits1_32_lo = _mmsi(and)(bits1, _mm(set1_epi32)(0xFFFF));
# ifdef __AVX2__
  auto bits0_32_hi =
      _mm(sllv_epi32)(_mm(srli_epi32)(bits0, 16), nbits0_32_lo);
  auto bits1_32_hi =
      _mm(sllv_epi32)(_mm(srli_epi32)(bits1, 16), nbits1_32_lo);
# else
  // emulate variable shift by abusing float exponents
  // first, convert to float
  auto bits0_32_hi = _mm_castps_si128(_mm(cvtepi32_ps)(_mm(srli_epi32)(bits0, 16)));
  auto bits1_32_hi = _mm_castps_si128(_mm(cvtepi32_ps)(_mm(srli_epi32)(bits1, 16)));
  
  // add shift amount to the exponent
  bits0_32_hi = _mm(add_epi32)(bits0_32_hi, _mm(slli_epi32)(nbits0_32_lo, 23));
  bits1_32_hi = _mm(add_epi32)(bits1_32_hi, _mm(slli_epi32)(nbits1_32_lo, 23));
  
  // convert back to int
  bits0_32_hi = _mm(cvtps_epi32)(_mm_castsi128_ps(bits0_32_hi));
  bits1_32_hi = _mm(cvtps_epi32)(_mm_castsi128_ps(bits1_32_hi));
# endif

  nbits_mix = _mm(madd_epi16)(nbits_mix, _mm(set1_epi16)(1));
  bits0 = _mmsi(or)(bits0_32_lo, bits0_32_hi);
  bits1 = _mmsi(or)(bits1_32_lo, bits1_32_hi);

  // 32 -> 64
# ifdef __AVX2__
  auto nbits_inv = _mm(subs_epu8)(_mm(set1_epi64x)(0x2020), nbits_mix);
  auto nbits0_64_lo = _mmsi(and)(nbits_inv, _mm(set1_epi32)(0xff));
  auto nbits1_64_lo = _mm(srli_epi16)(nbits_inv, 8);
  bits0 = _mm(sllv_epi32)(bits0, nbits0_64_lo);
  bits1 = _mm(sllv_epi32)(bits1, nbits1_64_lo);
  bits0 = _mm(srlv_epi64)(bits0, nbits0_64_lo);
  bits1 = _mm(srlv_epi64)(bits1, nbits1_64_lo);
# else
  auto nbits0_64_lo = _mmsi(and)(nbits_mix, _mm(set1_epi64x)(0xff));
  auto nbits1_64_lo = _mmsi(and)(_mm(srli_epi16)(nbits_mix, 8), _mm(set1_epi64x)(0xff));
  auto bits0_64_lo = _mmsi(and)(bits0, _mm(set1_epi64x)(0xFFFFFFFF));
  auto bits1_64_lo = _mmsi(and)(bits1, _mm(set1_epi64x)(0xFFFFFFFF));
  
  // just do two shifts for SSE variant
  auto bits0_64_hi = _mm(srli_epi64)(bits0, 32);
  auto bits1_64_hi = _mm(srli_epi64)(bits1, 32);
  
  bits0_64_hi = _mm_blend_epi16(
    _mm_sll_epi64(bits0_64_hi, nbits0_64_lo),
    _mm_sll_epi64(bits0_64_hi, _mm_unpackhi_epi64(nbits0_64_lo, nbits0_64_lo)),
    0xf0
  );
  bits1_64_hi = _mm_blend_epi16(
    _mm_sll_epi64(bits1_64_hi, nbits1_64_lo),
    _mm_sll_epi64(bits1_64_hi, _mm_unpackhi_epi64(nbits1_64_lo, nbits1_64_lo)),
    0xf0
  );
  
  bits0 = _mmsi(or)(bits0_64_lo, bits0_64_hi);
  bits1 = _mmsi(or)(bits1_64_lo, bits1_64_hi);
# endif

# ifdef __AVX2__
  auto bit_count = _mm_hadd_epi32(_mm256_castsi256_si128(nbits_mix), _mm256_extracti128_si256(nbits_mix, 1));
  bit_count = _mm_shuffle_epi8(bit_count, _mm_set_epi32(
    -1, -1, 0x0d090501, 0x0c080400
  ));
  alignas(16) uint8_t nbits_a[SIMD_WIDTH/4];
  _mm_storel_epi64((__m128i *)nbits_a, bit_count);
# else
  auto bit_count = _mm_hadd_epi32(nbits_mix, nbits_mix);
#  ifdef FPNGE_SLOW_PSHUFB
  bit_count = _mm_unpacklo_epi8(bit_count, _mm_srli_epi64(bit_count, 32));
#  else
  bit_count = _mm_shuffle_epi8(bit_count, _mm_set_epi32(
    -1, -1, 0x0d090c08, 0x05010400
  ));
#  endif
  alignas(16) uint8_t nbits_a[SIMD_WIDTH/2];
  _mm_storel_epi64((__m128i *)nbits_a, bit_count);
# endif

  // nbits_a <= 40 as we have at most 10 bits per symbol, so the call to the
  // writer is safe.
  alignas(SIMD_WIDTH) uint64_t bits_a[SIMD_WIDTH/4];
  _mmsi(store)((__mivec *)bits_a, bits0);
  _mmsi(store)((__mivec *)bits_a + 1, bits1);

#endif

  for (size_t ii = 0; ii < SIMD_WIDTH/4; ii++) {
    uint64_t bits = bits_a[kPerm[ii]];
#ifdef USE_PEXT
    bits = _pext_u64(bits, bitmask_a[kPerm[ii]]);
#endif
    writer->Write(nbits_a[kPerm[ii]], bits);
  }
}

// as above, but without high byte
static FORCE_INLINE void WriteBitsShort(__mivec nbits, __mivec bits,
                                 BitWriter *__restrict writer) {

#ifdef USE_PEXT
  // convert nbits into a mask
  auto bitmask = _mm(shuffle_epi8)(BCAST128(_mm_set_epi32(
    0xffffffff, 0xffffffff, 0x7f3f1f0f, 0x07030100
  )), nbits);
  auto bit_count = _mm(sad_epu8)(nbits, _mmsi(setzero)());
  
  alignas(SIMD_WIDTH) uint64_t nbits_a[SIMD_WIDTH/8];
  _mmsi(store)((__mivec *)nbits_a, bit_count);
  alignas(SIMD_WIDTH) uint64_t bits_a[SIMD_WIDTH/8];
  _mmsi(store)((__mivec *)bits_a, bits);
  alignas(SIMD_WIDTH) uint64_t bitmask_a[SIMD_WIDTH/8];
  _mmsi(store)((__mivec *)bitmask_a, bitmask);
#else
  // 8 -> 16
  auto prod = _mm(slli_epi16)(_mm(shuffle_epi8)(BCAST128(_mm_set_epi32(
    //0x80402010, 0x08040201, 0x80402010, 0x08040201
    // since we can't handle 8 bits, we'll under-shift it and do an extra shift
    -1, 0xffffff80, 0x40201008, 0x040201ff
  )), nbits), 8);
  auto bits_hi = _mm(mulhi_epu16)(_mmsi(andnot)(_mm(set1_epi16)(0xff), bits), prod);
  bits_hi = _mm(add_epi16)(bits_hi, bits_hi);
  bits = _mmsi(or)(_mmsi(and)(bits, _mm(set1_epi16)(0xff)), bits_hi);
  nbits = _mm(maddubs_epi16)(nbits, _mm(set1_epi8)(1));
  
  // 16 -> 32
  auto nbits_32_lo = _mmsi(and)(nbits, _mm(set1_epi32)(0xff));
  auto bits_32_lo = _mmsi(and)(bits, _mm(set1_epi32)(0xFFFF));
  auto bits_32_hi = _mm(srli_epi32)(bits, 16);
# ifdef __AVX2__
  bits_32_hi =
      _mm(sllv_epi32)(bits_32_hi, nbits_32_lo);
# else
  // need to avoid overflow when converting float -> int, because it converts to a signed int
  // do this by offsetting the shift by 1
  nbits_32_lo = _mm(sub_epi32)(nbits_32_lo, _mm(set1_epi32)(1));
  bits_32_hi = _mm_castps_si128(_mm(cvtepi32_ps)(bits_32_hi));
  bits_32_hi = _mm(add_epi32)(bits_32_hi, _mm(slli_epi32)(nbits_32_lo, 23));
  bits_32_hi = _mm(cvtps_epi32)(_mm_castsi128_ps(bits_32_hi));
  bits_32_hi = _mm(add_epi32)(bits_32_hi, bits_32_hi);
# endif
  nbits = _mm(madd_epi16)(nbits, _mm(set1_epi16)(1));
  bits = _mmsi(or)(bits_32_lo, bits_32_hi);

  // 32 -> 64
# ifdef __AVX2__
  auto nbits_inv = _mm(subs_epu8)(_mm(set1_epi64x)(0x20), nbits);
  bits = _mm(sllv_epi32)(bits, nbits_inv);
  bits = _mm(srlv_epi64)(bits, nbits_inv);
# else
  auto nbits_64_lo = _mmsi(and)(nbits, _mm(set1_epi64x)(0xff));
  auto bits_64_lo = _mmsi(and)(bits, _mm(set1_epi64x)(0xFFFFFFFF));
  auto bits_64_hi = _mm(srli_epi64)(bits, 32);
  bits_64_hi = _mm_blend_epi16(
    _mm_sll_epi64(bits_64_hi, nbits_64_lo),
    _mm_sll_epi64(bits_64_hi, _mm_unpackhi_epi64(nbits_64_lo, nbits_64_lo)),
    0xf0
  );
  bits = _mmsi(or)(bits_64_lo, bits_64_hi);
# endif
  
  auto nbits2 = _mm_hadd_epi32(
# ifdef __AVX2__
    _mm256_castsi256_si128(nbits), _mm256_extracti128_si256(nbits, 1)
# else
    nbits, nbits
# endif
  );
  
  alignas(16) uint32_t nbits_a[4];
  alignas(SIMD_WIDTH) uint64_t bits_a[SIMD_WIDTH/8];
  
  /*
# ifdef __AVX2__
  auto long_symbols = _mm_movemask_epi8(_mm_cmpgt_epi8(nbits2, _mm_set1_epi8(28)));
  if (long_symbols == 0) {
    // if top half of each 64-bit group is empty, we can do one more combine round - hopefully we arrive here often
    auto bits2 = _mm_castps_si128(_mm_shuffle_ps(
      _mm_castsi128_ps(_mm256_castsi256_si128(bits)), _mm_castsi128_ps(_mm256_extracti128_si256(bits, 1)),
      _MM_SHUFFLE(2, 0, 2, 0)));
    
    auto nbits_shift = _mm_subs_epu8(_mm_set1_epi64x(32), nbits2);
    bits2 = _mm_sllv_epi32(bits2, nbits_shift);
    bits2 = _mm_srlv_epi64(bits2, nbits_shift);
    
    _mm_store_si128((__m128i *)bits_a, bits2);
    
    nbits2 = _mm_hadd_epi32(nbits2, nbits2);
    _mm_storel_epi64((__m128i *)nbits_a, nbits2);
    
    writer->Write(nbits_a[0], bits_a[0]);
    writer->Write(nbits_a[1], bits_a[1]);
    return;
  }
# endif
  */
  
  _mm_store_si128((__m128i *)nbits_a, nbits2);
  _mmsi(store)((__mivec *)bits_a, bits);

#endif

  for (size_t ii = 0; ii < SIMD_WIDTH/8; ii++) {
    uint64_t bits = bits_a[ii];
#ifdef USE_PEXT
    bits = _pext_u64(bits, bitmask_a[ii]);
#endif
    if (nbits_a[ii] + writer->bits_in_buffer > 63) {
      // hope this case rarely occurs
      writer->Write(16, bits & 0xffff);
      bits >>= 16;
      nbits_a[ii] -= 16;
    }
    writer->Write(nbits_a[ii], bits);
  }
}

static void EncodeOneRow(size_t bytes_per_line,
                  const unsigned char *current_row_buf, const unsigned char *top_buf,
                  const unsigned char *left_buf, const unsigned char *topleft_buf,
                  const HuffmanTable &table,
                  uint32_t &s1, uint32_t &s2, size_t dist_nbits,
                  size_t dist_bits, BitWriter *__restrict writer) {
#ifndef FPNGE_FIXED_PREDICTOR
  uint8_t predictor;
  size_t best_cost = ~0U;
  TryPredictor<1>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, table, best_cost, predictor, dist_nbits);
  TryPredictor<2>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, table, best_cost, predictor, dist_nbits);
  TryPredictor<3>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, table, best_cost, predictor, dist_nbits);
  TryPredictor<4>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, table, best_cost, predictor, dist_nbits);
#else
  uint8_t predictor = FPNGE_FIXED_PREDICTOR;
#endif

  writer->Write(table.first16_nbits[predictor], table.first16_bits[predictor]);
  UpdateAdler32(s1, s2, predictor);

  auto adler_accum_s1 = INT2VEC(s1);
  auto adler_accum_s2 = INT2VEC(s2);
  auto adler_s1_sum = _mmsi(setzero)();

  uint16_t bytes_since_flush = 1;

  auto flush_adler = [&]() {
    adler_accum_s2 = _mm(add_epi32)(adler_accum_s2,
                                    _mm(slli_epi32)(adler_s1_sum, SIMD_WIDTH == 32 ? 5 : 4));
    adler_s1_sum = _mmsi(setzero)();
    
    uint32_t ls1 = hadd(adler_accum_s1);
    uint32_t ls2 = hadd(adler_accum_s2);
    ls1 %= kAdler32Mod;
    ls2 %= kAdler32Mod;
    s1 = ls1;
    s2 = ls2;
    adler_accum_s1 = INT2VEC(s1);
    adler_accum_s2 = INT2VEC(s2);
    bytes_since_flush = 0;
  };

  auto encode_chunk_cb = [&](const __mivec bytes,
                             const size_t bytes_in_vec) FORCE_INLINE_LAMBDA {
    auto maskv = _mmsi(loadu)((__mivec *)(kMaskVec - bytes_in_vec));

    auto data_for_lut = _mmsi(and)(_mm(set1_epi8)(0xF), bytes);
    auto use_mid = _mm(cmpgt_epi8)(
      _mm(set1_epi8)(96),
      _mm(add_epi8)(bytes, _mm(set1_epi8)(112))
    );
    auto nbits_low16 =
        _mm(shuffle_epi8)(BCAST128(
                                _mm_load_si128((__m128i *)table.first16_nbits)),
                            data_for_lut);
    auto nbits_hi16 =
        _mm(shuffle_epi8)(BCAST128(
                                _mm_load_si128((__m128i *)table.last16_nbits)),
                            data_for_lut);

    auto bits_low16 =
        _mm(shuffle_epi8)(BCAST128(
                                _mm_load_si128((__m128i *)table.first16_bits)),
                            data_for_lut);
    auto bits_hi16 =
        _mm(shuffle_epi8)(BCAST128(
                                _mm_load_si128((__m128i *)table.last16_bits)),
                            data_for_lut);
    
    if (_mm(movemask_epi8)(use_mid)) {
      auto data_for_midlut =
          _mmsi(and)(_mm(set1_epi8)(0xF), _mm(srai_epi16)(bytes, 4));
      
      auto bits_mid_lo =
          _mm(shuffle_epi8)(BCAST128(
                                  _mm_load_si128((__m128i *)table.mid_lowbits)),
                              data_for_midlut);

      auto bits_mid_hi =
          _mm(shuffle_epi8)(BCAST128(_mm_load_si128(
                                  (__m128i *)kBitReverseNibbleLookup)),
                              data_for_lut);

      auto nbits = _mm(blendv_epi8)(
        _mm(blendv_epi8)(nbits_low16, nbits_hi16, bytes),
        _mm(set1_epi8)(table.mid_nbits),
        use_mid
      );
      nbits = _mmsi(and)(nbits, maskv);

      auto bits_lo = _mm(blendv_epi8)(
        _mm(blendv_epi8)(bits_low16, bits_hi16, bytes),
        bits_mid_lo,
        use_mid
      );

      // need to mask out unused hi bits because WriteBits uses OR operations to merge data (this might not be needed in PEXT variant)
      auto bits_hi = _mmsi(and)(use_mid, bits_mid_hi);
      bits_lo = _mmsi(and)(bits_lo, maskv);
      bits_hi = _mmsi(and)(bits_hi, maskv);

      WriteBits(nbits, bits_lo, bits_hi, table.mid_nbits - 4, writer);
    } else {
      // mid never used - we can make some shortcuts
      auto nbits = _mm(blendv_epi8)(nbits_low16, nbits_hi16, bytes);
      auto bits_lo = _mm(blendv_epi8)(bits_low16, bits_hi16, bytes);
      nbits = _mmsi(and)(nbits, maskv);
      bits_lo = _mmsi(and)(bits_lo, maskv);
      
      WriteBitsShort(nbits, bits_lo, writer);
    }
  };

  auto adler_chunk_cb = [&](const __mivec pdata, const size_t bytes_in_vec) FORCE_INLINE_LAMBDA {
    bytes_since_flush += bytes_in_vec;
    auto bytes = pdata;
    
    if (bytes_in_vec < SIMD_WIDTH) {
      adler_accum_s2 = _mm(add_epi32)(
          _mm(mul_epu32)(_mm(set1_epi32)(bytes_in_vec), adler_accum_s1),
          adler_accum_s2);
      bytes = _mmsi(and)(bytes, _mmsi(loadu)((__mivec *)(kMaskVec - bytes_in_vec)));
    } else {
      adler_s1_sum = _mm(add_epi32)(adler_s1_sum, adler_accum_s1);
    }

    adler_accum_s1 = _mm(add_epi32)(
        adler_accum_s1, _mm(sad_epu8)(bytes, _mmsi(setzero)()));

    auto muls = _mm(set_epi8)(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
#if SIMD_WIDTH == 32
      , 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
#endif
    );
    auto bytesmuls = _mm(maddubs_epi16)(bytes, muls);
    adler_accum_s2 = _mm(add_epi32)(
        adler_accum_s2, _mm(madd_epi16)(bytesmuls, _mm(set1_epi16)(1)));

    if (bytes_since_flush >= 5500) {
      flush_adler();
    }
  };

  auto encode_rle_cb = [&](size_t run) {
    writer->Write(table.first16_nbits[0], table.first16_bits[0]);
    ForAllRLESymbols(run, [&](size_t len, size_t count) {
      uint32_t bits = (dist_bits << table.lz77_length_nbits[len]) | table.lz77_length_bits[len];
      auto nbits = table.lz77_length_nbits[len] + dist_nbits;
      while (count--) {
        writer->Write(nbits, bits);
      }
    });
  };

#ifdef FPNGE_FIXED_PREDICTOR
  if (predictor == 0) {
    ProcessRow<0>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, encode_chunk_cb, adler_chunk_cb, encode_rle_cb);
  } else
#endif
  if (predictor == 1) {
    ProcessRow<1>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, encode_chunk_cb, adler_chunk_cb, encode_rle_cb);
  } else if (predictor == 2) {
    ProcessRow<2>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, encode_chunk_cb, adler_chunk_cb, encode_rle_cb);
  } else if (predictor == 3) {
    ProcessRow<3>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, encode_chunk_cb, adler_chunk_cb, encode_rle_cb);
  } else {
    ProcessRow<4>(bytes_per_line, current_row_buf, top_buf, left_buf,
                  topleft_buf, encode_chunk_cb, adler_chunk_cb, encode_rle_cb);
  }

  flush_adler();
}

static void CollectSymbolCounts(
    size_t bytes_per_line,
    const unsigned char *current_row_buf, const unsigned char *top_buf,
    const unsigned char *left_buf, const unsigned char *topleft_buf,
    uint64_t *__restrict symbol_counts) {

  auto encode_chunk_cb = [&](const __mivec pdata,
                             const size_t bytes_in_vec) {
    alignas(SIMD_WIDTH) uint8_t predicted_data[SIMD_WIDTH];
    _mmsi(store)((__mivec *)predicted_data, pdata);
    for (size_t i = 0; i < bytes_in_vec; i++) {
      symbol_counts[predicted_data[i]] += 1;
    }
  };

  auto adler_chunk_cb = [&](const __mivec, const size_t) {
  };

  auto encode_rle_cb = [&](size_t run) {
    symbol_counts[0] += 1;
    constexpr size_t kLZ77Sym[] = {
        0,   0,   0,   257, 258, 259, 260, 261, 262, 263, 264, 265, 265, 266,
        266, 267, 267, 268, 268, 269, 269, 269, 269, 270, 270, 270, 270, 271,
        271, 271, 271, 272, 272, 272, 272, 273, 273, 273, 273, 273, 273, 273,
        273, 274, 274, 274, 274, 274, 274, 274, 274, 275, 275, 275, 275, 275,
        275, 275, 275, 276, 276, 276, 276, 276, 276, 276, 276, 277, 277, 277,
        277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 277, 278,
        278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278, 278,
        278, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279, 279,
        279, 279, 279, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280, 280,
        280, 280, 280, 280, 280, 281, 281, 281, 281, 281, 281, 281, 281, 281,
        281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281, 281,
        281, 281, 281, 281, 281, 281, 281, 281, 281, 282, 282, 282, 282, 282,
        282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282,
        282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 282, 283,
        283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283,
        283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283, 283,
        283, 283, 283, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284,
        284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284, 284,
        284, 284, 284, 284, 284, 284, 285,
    };
    ForAllRLESymbols(run,
                     [&](size_t len, size_t count) { symbol_counts[kLZ77Sym[len]] += count; });
  };

#ifdef FPNGE_FIXED_PREDICTOR
  ProcessRow<FPNGE_FIXED_PREDICTOR>(
      bytes_per_line, current_row_buf, top_buf, left_buf, topleft_buf,
      encode_chunk_cb, adler_chunk_cb, encode_rle_cb);
#else
  ProcessRow<4>(bytes_per_line, current_row_buf, top_buf, left_buf,
                topleft_buf, encode_chunk_cb, adler_chunk_cb, encode_rle_cb);
#endif
}

static void AppendBE32(size_t value, BitWriter *__restrict writer) {
  writer->Write(8, value >> 24);
  writer->Write(8, (value >> 16) & 0xFF);
  writer->Write(8, (value >> 8) & 0xFF);
  writer->Write(8, value & 0xFF);
}

static void WriteHeader(size_t width, size_t height, size_t bytes_per_channel,
                 size_t num_channels, BitWriter *__restrict writer) {
  constexpr uint8_t kPNGHeader[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  for (size_t i = 0; i < 8; i++) {
    writer->Write(8, kPNGHeader[i]);
  }
  // Length
  writer->Write(32, 0x0d000000);
  assert(writer->bits_in_buffer == 0);
  size_t crc_start = writer->bytes_written;
  // IHDR
  writer->Write(32, 0x52444849);
  AppendBE32(width, writer);
  AppendBE32(height, writer);
  // Bit depth
  writer->Write(8, bytes_per_channel * 8);
  // Colour type
  constexpr uint8_t numc_to_colour_type[] = {0, 0, 4, 2, 6};
  writer->Write(8, numc_to_colour_type[num_channels]);
  // Compression, filter and interlace methods.
  writer->Write(24, 0);
  assert(writer->bits_in_buffer == 0);
  size_t crc_end = writer->bytes_written;
  uint32_t crc = compute_crc(writer->data + crc_start, crc_end - crc_start);
  AppendBE32(crc, writer);
}

size_t FPNGEEncode(size_t bytes_per_channel, size_t num_channels,
                   const unsigned char *data, size_t width, size_t row_stride,
                   size_t height, unsigned char **output) {
  assert(bytes_per_channel == 1 || bytes_per_channel == 2);
  assert(num_channels != 0 && num_channels <= 4);
  size_t bytes_per_line = bytes_per_channel * num_channels * width;
  assert(row_stride >= bytes_per_line);

  // allows for padding, and for extra initial space for the "left" pixel for
  // predictors.
  size_t bytes_per_line_buf =
      (bytes_per_line + 4 * bytes_per_channel + SIMD_WIDTH-1) / SIMD_WIDTH * SIMD_WIDTH;

  // Extra space for alignment purposes.
  std::vector<unsigned char> buf(bytes_per_line_buf * 2 + SIMD_WIDTH-1 +
                                 4 * bytes_per_channel);
  unsigned char *aligned_buf_ptr = buf.data() + 4 * bytes_per_channel;
  aligned_buf_ptr += (intptr_t)aligned_buf_ptr % SIMD_WIDTH
                         ? (SIMD_WIDTH - (intptr_t)aligned_buf_ptr % SIMD_WIDTH)
                         : 0;

  // likely an overestimate
  *output = (unsigned char *)malloc(1024 + 2 * bytes_per_line * height);

  BitWriter writer;
  writer.data = *output;

  WriteHeader(width, height, bytes_per_channel, num_channels, &writer);

  assert(writer.bits_in_buffer == 0);
  size_t chunk_length_pos = writer.bytes_written;
  writer.bytes_written += 4; // Skip space for length.
  size_t crc_pos = writer.bytes_written;
  writer.Write(32, 0x54414449); // IDAT
  // Deflate header
  writer.Write(8, 8);  // deflate with smallest window
  writer.Write(8, 29); // cfm+flg check value

  uint64_t symbol_counts[286] = {};

  // Sample ~1.5% of the rows in the center of the image.
  size_t y0 = height * 126 / 256;
  size_t y1 = height * 130 / 256;

  for (size_t y = y0; y < y1; y++) {
    const unsigned char *current_row_in = data + row_stride * y;
    unsigned char *current_row_buf =
        aligned_buf_ptr + (y % 2 ? bytes_per_line_buf : 0);
    const unsigned char *top_buf =
        aligned_buf_ptr + ((y + 1) % 2 ? bytes_per_line_buf : 0);
    const unsigned char *left_buf =
        current_row_buf - bytes_per_channel * num_channels;
    const unsigned char *topleft_buf =
        top_buf - bytes_per_channel * num_channels;

    memcpy(current_row_buf, current_row_in, bytes_per_line);
    if (y == y0) {
      continue;
    }

    CollectSymbolCounts(bytes_per_line,
                        current_row_buf, top_buf,
                        left_buf, topleft_buf, symbol_counts);
  }

  memset(buf.data(), 0, buf.size());

  HuffmanTable huffman_table(symbol_counts);

  // Single block, dynamic huffman
  writer.Write(3, 0b101);
  uint32_t dist_nbits;
  uint32_t dist_bits;
  WriteHuffmanCode(dist_nbits, dist_bits, huffman_table, &writer);

  uint32_t crc = ~0U;
  uint32_t s1 = 1;
  uint32_t s2 = 0;
  for (size_t y = 0; y < height; y++) {
    const unsigned char *current_row_in = data + row_stride * y;
    unsigned char *current_row_buf =
        aligned_buf_ptr + (y % 2 ? bytes_per_line_buf : 0);
    const unsigned char *top_buf =
        aligned_buf_ptr + ((y + 1) % 2 ? bytes_per_line_buf : 0);
    const unsigned char *left_buf =
        current_row_buf - bytes_per_channel * num_channels;
    const unsigned char *topleft_buf =
        top_buf - bytes_per_channel * num_channels;

    memcpy(current_row_buf, current_row_in, bytes_per_line);

    EncodeOneRow(bytes_per_line,
                 current_row_buf, top_buf, left_buf,
                 topleft_buf, huffman_table, s1, s2, dist_nbits, dist_bits,
                 &writer);

    size_t bytes = (writer.bytes_written - crc_pos) / 64 * 64;
    crc = update_crc(crc, writer.data + crc_pos, bytes);
    crc_pos += bytes;
  }

  // EOB
  writer.Write(huffman_table.nbits[256], huffman_table.end_bits);

  writer.ZeroPadToByte();
  assert(writer.bits_in_buffer == 0);
  s1 %= kAdler32Mod;
  s2 %= kAdler32Mod;
  uint32_t adler32 = (s2 << 16) | s1;
  AppendBE32(adler32, &writer);

  size_t data_len = writer.bytes_written - chunk_length_pos - 8;
  writer.data[chunk_length_pos + 0] = data_len >> 24;
  writer.data[chunk_length_pos + 1] = (data_len >> 16) & 0xFF;
  writer.data[chunk_length_pos + 2] = (data_len >> 8) & 0xFF;
  writer.data[chunk_length_pos + 3] = data_len & 0xFF;

  crc = update_crc(crc, writer.data + crc_pos, writer.bytes_written - crc_pos);
  crc_pos = writer.bytes_written;
  crc ^= ~0U;
  AppendBE32(crc, &writer);

  // IEND
  writer.Write(32, 0);
  writer.Write(32, 0x444e4549);
  writer.Write(32, 0x826042ae);

  return writer.bytes_written;
}
