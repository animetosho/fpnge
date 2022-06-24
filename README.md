# Fast PNG Encoder
This is a fork of the excellent [fpnge](https://github.com/veluca93/fpnge) PNG encoder.

This fork introduces the following to the original project:

* Support for SSE4.1 instead of AVX2, to support a wider range of CPUs
  * Currently PCLMULQDQ is still required until fallback CRC32 is added
* Around 50% faster performance (under AVX2)
* API changed slightly to allow custom memory allocators

Limitations:

* Only compiles for x86 CPUs with SSE4.1+PCLMUL minimum
* No runtime CPU detection available (relies on compiler targeting, i.e. `-march=native`)
* Doesn’t support indexed colour or bit depths below 8 bits

## Original README

This is a proof-of-concept fast PNG encoder that uses AVX2 and a special
Huffman table to encode images faster. Speed on a single core is anywhere from
180 to 800 MP/s on a Threadripper 3970x, depending on compile time settings and
content.

At the moment, only RGB(A) input is supported.