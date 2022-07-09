
#include "crc32.h"
#include <cassert>
#include <nmmintrin.h>
#include <wmmintrin.h>

Crc32Clmul::Crc32Clmul() {
  auto s = (__m128i *)state;
  s[0] = _mm_cvtsi32_si128(0x9db42487);
  s[1] = _mm_setzero_si128();
  s[2] = _mm_setzero_si128();
  s[3] = _mm_setzero_si128();
}

#ifdef __AVX512VL__
static inline __m128i do_one_fold_merge(__m128i src, __m128i data) {
  const auto k = _mm_set_epi32(0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596);
  return _mm_ternarylogic_epi32(_mm_clmulepi64_si128(src, k, 0x01),
                                _mm_clmulepi64_si128(src, k, 0x10), data, 0x96);
}
static inline __m128i double_xor(__m128i a, __m128i b, __m128i c) {
  return _mm_ternarylogic_epi32(a, b, c, 0x96);
}
#else
static inline __m128i do_one_fold(__m128i src) {
  const auto k = _mm_set_epi32(0x00000001, 0x54442bd4, 0x00000001, 0xc6e41596);
  return _mm_xor_si128(_mm_clmulepi64_si128(src, k, 0x01),
                       _mm_clmulepi64_si128(src, k, 0x10));
}
static inline __m128i do_one_fold_merge(__m128i src, __m128i data) {
  return _mm_xor_si128(do_one_fold(src), data);
}
static inline __m128i double_xor(__m128i a, __m128i b, __m128i c) {
  a = _mm_xor_si128(a, b);
  return _mm_xor_si128(a, c);
}
#endif

void Crc32Clmul::update64(const void *buf, size_t len) {
  assert((len & 63) == 0);
  auto s = (__m128i *)state;
  auto x0 = _mm_loadu_si128(s + 0);
  auto x1 = _mm_loadu_si128(s + 1);
  auto x2 = _mm_loadu_si128(s + 2);
  auto x3 = _mm_loadu_si128(s + 3);

  auto src = (const __m128i *)buf;

  while (len) {
#ifdef __AVX512VL__
    x0 = do_one_fold_merge(x0, _mm_loadu_si128(src + 0));
    x1 = do_one_fold_merge(x1, _mm_loadu_si128(src + 1));
    x2 = do_one_fold_merge(x2, _mm_loadu_si128(src + 2));
    x3 = do_one_fold_merge(x3, _mm_loadu_si128(src + 3));
#else
    // nesting do_one_fold() in _mm_xor_si128() seems to cause MSVC to generate
    // horrible code, so separate it out
    x0 = do_one_fold(x0);
    x1 = do_one_fold(x1);
    x2 = do_one_fold(x2);
    x3 = do_one_fold(x3);
    x0 = _mm_xor_si128(x0, _mm_loadu_si128(src + 0));
    x1 = _mm_xor_si128(x1, _mm_loadu_si128(src + 1));
    x2 = _mm_xor_si128(x2, _mm_loadu_si128(src + 2));
    x3 = _mm_xor_si128(x3, _mm_loadu_si128(src + 3));
#endif
  }
  _mm_storeu_si128(s + 0, x0);
  _mm_storeu_si128(s + 1, x1);
  _mm_storeu_si128(s + 2, x2);
  _mm_storeu_si128(s + 3, x3);
}

alignas(16) static const unsigned pshufb_shf_table[60] = {
    0x84838281, 0x88878685, 0x8c8b8a89, 0x008f8e8d, /* shl 15 (16 - 1)/shr1 */
    0x85848382, 0x89888786, 0x8d8c8b8a, 0x01008f8e, /* shl 14 (16 - 3)/shr2 */
    0x86858483, 0x8a898887, 0x8e8d8c8b, 0x0201008f, /* shl 13 (16 - 4)/shr3 */
    0x87868584, 0x8b8a8988, 0x8f8e8d8c, 0x03020100, /* shl 12 (16 - 4)/shr4 */
    0x88878685, 0x8c8b8a89, 0x008f8e8d, 0x04030201, /* shl 11 (16 - 5)/shr5 */
    0x89888786, 0x8d8c8b8a, 0x01008f8e, 0x05040302, /* shl 10 (16 - 6)/shr6 */
    0x8a898887, 0x8e8d8c8b, 0x0201008f, 0x06050403, /* shl  9 (16 - 7)/shr7 */
    0x8b8a8988, 0x8f8e8d8c, 0x03020100, 0x07060504, /* shl  8 (16 - 8)/shr8 */
    0x8c8b8a89, 0x008f8e8d, 0x04030201, 0x08070605, /* shl  7 (16 - 9)/shr9 */
    0x8d8c8b8a, 0x01008f8e, 0x05040302, 0x09080706, /* shl  6 (16 -10)/shr10*/
    0x8e8d8c8b, 0x0201008f, 0x06050403, 0x0a090807, /* shl  5 (16 -11)/shr11*/
    0x8f8e8d8c, 0x03020100, 0x07060504, 0x0b0a0908, /* shl  4 (16 -12)/shr12*/
    0x008f8e8d, 0x04030201, 0x08070605, 0x0c0b0a09, /* shl  3 (16 -13)/shr13*/
    0x01008f8e, 0x05040302, 0x09080706, 0x0d0c0b0a, /* shl  2 (16 -14)/shr14*/
    0x0201008f, 0x06050403, 0x0a090807, 0x0e0d0c0b  /* shl  1 (16 -15)/shr15*/
};

uint32_t Crc32Clmul::update_final(const void *buf, size_t len) {
  if (len >= 64) {
    update64(buf, len & ~63);
    buf = (const char *)buf + (len & ~63);
    len &= 63;
  }

  __m128i x0, x1, x2, x3;
  __m128i crc_fold;
  auto src = (uint8_t *)buf;
  auto crc = (__m128i *)state;

  if (len >= 48) {
    x0 = _mm_loadu_si128((__m128i *)src);
    x1 = _mm_loadu_si128((__m128i *)src + 1);
    x2 = _mm_loadu_si128((__m128i *)src + 2);

    x3 = crc[3];
    crc[3] = do_one_fold_merge(crc[2], x2);
    crc[2] = do_one_fold_merge(crc[1], x1);
    crc[1] = do_one_fold_merge(crc[0], x0);
    crc[0] = x3;
  } else if (len >= 32) {
    x0 = _mm_loadu_si128((__m128i *)src);
    x1 = _mm_loadu_si128((__m128i *)src + 1);

    x2 = crc[2];
    x3 = crc[3];
    crc[3] = do_one_fold_merge(crc[1], x1);
    crc[2] = do_one_fold_merge(crc[0], x0);
    crc[1] = x3;
    crc[0] = x2;
  } else if (len >= 16) {
    x0 = _mm_loadu_si128((__m128i *)src);

    x3 = crc[3];
    crc[3] = do_one_fold_merge(crc[0], x0);
    crc[0] = crc[1];
    crc[1] = crc[2];
    crc[2] = x3;
  }
  src += (len & 48);
  len &= 15;

  if (len > 0) {
    auto xmm_shl = _mm_load_si128((__m128i *)pshufb_shf_table + (len - 1));
    auto xmm_shr = _mm_xor_si128(xmm_shl, _mm_set1_epi8(-128));

    x0 = _mm_loadu_si128((__m128i *)src);
    x1 = _mm_shuffle_epi8(crc[0], xmm_shl);

    crc[0] = _mm_or_si128(_mm_shuffle_epi8(crc[0], xmm_shr),
                          _mm_shuffle_epi8(crc[1], xmm_shl));
    crc[1] = _mm_or_si128(_mm_shuffle_epi8(crc[1], xmm_shr),
                          _mm_shuffle_epi8(crc[2], xmm_shl));
    crc[2] = _mm_or_si128(_mm_shuffle_epi8(crc[2], xmm_shr),
                          _mm_shuffle_epi8(crc[3], xmm_shl));
    crc[3] = _mm_or_si128(_mm_shuffle_epi8(crc[3], xmm_shr),
                          _mm_shuffle_epi8(x0, xmm_shl));

    crc[3] = do_one_fold_merge(x1, crc[3]);
  }

  crc_fold = _mm_set_epi32(0x00000001, 0x751997d0, // rk2
                           0x00000000, 0xccaa009e  // rk1
  );

  x0 = double_xor(crc[1], _mm_clmulepi64_si128(crc[0], crc_fold, 0x10),
                  _mm_clmulepi64_si128(crc[0], crc_fold, 0x01));
  x0 = double_xor(crc[2], _mm_clmulepi64_si128(x0, crc_fold, 0x10),
                  _mm_clmulepi64_si128(x0, crc_fold, 0x01));
  x0 = double_xor(crc[3], _mm_clmulepi64_si128(x0, crc_fold, 0x10),
                  _mm_clmulepi64_si128(x0, crc_fold, 0x01));

  crc_fold = _mm_set_epi32(0x00000001, 0x63cd6124, // rk6
                           0x00000000, 0xccaa009e  // rk5 / rk1
  );

  x1 = _mm_xor_si128(_mm_clmulepi64_si128(x0, crc_fold, 0),
                     _mm_srli_si128(x0, 8));

  x0 = _mm_slli_si128(x1, 4);
  x0 = _mm_clmulepi64_si128(x0, crc_fold, 0x10);
#ifdef __AVX512VL__
  x0 = _mm_ternarylogic_epi32(x0, x1, _mm_set_epi32(0, -1, -1, 0), 0x28);
#else
  x1 = _mm_and_si128(x1, _mm_set_epi32(0, -1, -1, 0));
  x0 = _mm_xor_si128(x0, x1);
#endif

  crc_fold = _mm_set_epi32(0x00000001, 0xdb710640, // rk8
                           0x00000000, 0xf7011641  // rk7
  );

  x1 = _mm_clmulepi64_si128(x0, crc_fold, 0);
  x1 = _mm_clmulepi64_si128(x1, crc_fold, 0x10);
#ifdef __AVX512VL__
  x1 = _mm_ternarylogic_epi32(x1, x0, x0, 0xC3); // NOT(XOR(t1, t0))
#else
  x0 = _mm_xor_si128(x0, _mm_set_epi32(0, -1, -1, 0));
  x1 = _mm_xor_si128(x1, x0);
#endif
  return _mm_extract_epi32(x1, 2);
}
