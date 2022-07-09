#include <stdint.h>
#include <stdlib.h>

class ICrc32 {
public:
  virtual uint32_t update_final(const void *data, size_t len) = 0;
  virtual void update64(const void *data, size_t len) = 0;
};

class Crc32Clmul : public ICrc32 {
  uint32_t state[16];

public:
  Crc32Clmul();
  static inline ICrc32 *create() { return new Crc32Clmul(); }
  void update64(const void *data, size_t len) override;
  uint32_t update_final(const void *data, size_t len) override;
};

class Crc32Slicing : public ICrc32 {
  uint32_t state;

public:
  Crc32Slicing();
  static inline ICrc32 *create() { return new Crc32Slicing(); }
  void update64(const void *data, size_t len) override;
  uint32_t update_final(const void *data, size_t len) override;
};
