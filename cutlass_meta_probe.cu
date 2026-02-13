#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "cutlass/layout/matrix.h"

constexpr int M = 16;
constexpr int K = 32;
constexpr int K_GROUP = 4;
constexpr int PACKED_K = K / 2;

static inline uint32_t encode_2of4(int i0, int i1) {
  if (i0 == 0 && i1 == 1) return 0;
  if (i0 == 0 && i1 == 2) return 1;
  if (i0 == 0 && i1 == 3) return 2;
  if (i0 == 1 && i1 == 2) return 3;
  if (i0 == 1 && i1 == 3) return 4;
  if (i0 == 2 && i1 == 3) return 5;
  return 0;
}

static inline void decode_2of4(uint32_t code, int &i0, int &i1) {
  switch (code & 0x7u) {
    case 0: i0 = 0; i1 = 1; break;
    case 1: i0 = 0; i1 = 2; break;
    case 2: i0 = 0; i1 = 3; break;
    case 3: i0 = 1; i1 = 2; break;
    case 4: i0 = 1; i1 = 3; break;
    case 5: i0 = 2; i1 = 3; break;
    default: i0 = 0; i1 = 1; break;
  }
}

// Host encoder for CUTLASS-style metadata layout
// - Ordered 2-bit indices per logical element
// - Packed per row-pair into two 32-bit chunks (0..7, 8..15)
// - Stored with ColumnMajorInterleaved<2> layout
static void encode_cutlass_metadata_interleaved2(
  const std::vector<uint32_t> &meta_uncompressed,
  std::vector<uint32_t> &meta_out) {
  using LayoutE = cutlass::layout::ColumnMajorInterleaved<2>;
  LayoutE layout(LayoutE::packed({M * 2, 1}).stride());

  meta_out.assign(M * 4, 0u); // 32 logical rows, interleave=2 => 64 u32 slots

  auto offset = [&](int logical_row) {
    return layout({logical_row, 0});
  };

  for (int rp = 0; rp < M / 2; ++rp) {
    int row0 = rp * 2;
    int row1 = row0 + 1;

    uint32_t chunk0_r0 = 0, chunk1_r0 = 0;
    uint32_t chunk0_r1 = 0, chunk1_r1 = 0;

    for (int i = 0; i < 8; ++i) {
      chunk0_r0 |= (meta_uncompressed[row0 * PACKED_K + i] & 0x3u) << (2 * i);
      chunk1_r0 |= (meta_uncompressed[row0 * PACKED_K + (i + 8)] & 0x3u) << (2 * i);
      chunk0_r1 |= (meta_uncompressed[row1 * PACKED_K + i] & 0x3u) << (2 * i);
      chunk1_r1 |= (meta_uncompressed[row1 * PACKED_K + (i + 8)] & 0x3u) << (2 * i);
    }

    uint32_t packed0 = chunk0_r0 | (chunk0_r1 << 16);
    uint32_t packed1 = chunk1_r0 | (chunk1_r1 << 16);

    int logical_row0 = rp * 2 + 0;
    int logical_row1 = rp * 2 + 1;
    meta_out[offset(logical_row0)] = packed0;
    meta_out[offset(logical_row1)] = packed1;
  }
}

static void decode_cutlass_metadata_interleaved2(
    const std::vector<uint32_t> &meta_in,
    std::vector<uint32_t> &meta_uncompressed_out) {
  using LayoutE = cutlass::layout::ColumnMajorInterleaved<2>;
  LayoutE layout(LayoutE::packed({M * 2, 1}).stride());

  auto offset = [&](int logical_row) {
    return layout({logical_row, 0});
  };

  meta_uncompressed_out.assign(M * PACKED_K, 0u);

  for (int rp = 0; rp < M / 2; ++rp) {
    int logical_row0 = rp * 2 + 0;
    int logical_row1 = rp * 2 + 1;

    uint32_t packed0 = meta_in[offset(logical_row0)];
    uint32_t packed1 = meta_in[offset(logical_row1)];

    uint32_t chunk0_r0 = packed0 & 0xFFFFu;
    uint32_t chunk0_r1 = (packed0 >> 16) & 0xFFFFu;
    uint32_t chunk1_r0 = packed1 & 0xFFFFu;
    uint32_t chunk1_r1 = (packed1 >> 16) & 0xFFFFu;

    int row0 = rp * 2;
    int row1 = row0 + 1;

    for (int i = 0; i < 8; ++i) {
      meta_uncompressed_out[row0 * PACKED_K + i] = (chunk0_r0 >> (2 * i)) & 0x3u;
      meta_uncompressed_out[row0 * PACKED_K + (i + 8)] = (chunk1_r0 >> (2 * i)) & 0x3u;
      meta_uncompressed_out[row1 * PACKED_K + i] = (chunk0_r1 >> (2 * i)) & 0x3u;
      meta_uncompressed_out[row1 * PACKED_K + (i + 8)] = (chunk1_r1 >> (2 * i)) & 0x3u;
    }
  }
}

// Offline init: dense -> 2:4 sparse (top-2 per group)
static void prune_dense_to_sparse_2of4(
    const std::vector<__half> &A_dense,
    std::vector<__half> &A_sparse_out) {
  A_sparse_out = A_dense;
  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < K / K_GROUP; ++g) {
      float vals[4];
      int idx[4] = {0, 1, 2, 3};
      for (int i = 0; i < 4; ++i) {
        vals[i] = std::abs(__half2float(A_sparse_out[m * K + g * K_GROUP + i]));
      }
      std::partial_sort(idx, idx + 2, idx + 4,
                        [&](int a, int b) { return vals[a] > vals[b]; });
      int i0 = idx[0];
      int i1 = idx[1];
      for (int i = 0; i < 4; ++i) {
        if (i != i0 && i != i1) {
          A_sparse_out[m * K + g * K_GROUP + i] = __float2half(0.0f);
        }
      }
    }
  }
}

// Encode from already-pruned sparse A (2 nonzeros per 4). No top-k selection.
static bool encode_sparse_to_packed_and_meta(
    const std::vector<__half> &A_sparse,
    std::vector<__half> &A_packed,
    std::vector<uint32_t> &meta_uncompressed) {
  A_packed.assign(M * PACKED_K, __float2half(0.0f));
  meta_uncompressed.assign(M * PACKED_K, 0u);

  for (int m = 0; m < M; ++m) {
    for (int g = 0; g < K / K_GROUP; ++g) {
      int nz_idx[2] = {-1, -1};
      int found = 0;
      for (int i = 0; i < 4; ++i) {
        float v = __half2float(A_sparse[m * K + g * K_GROUP + i]);
        if (v != 0.0f) {
          if (found < 2) nz_idx[found] = i;
          ++found;
        }
      }

      if (found != 2) {
        return false;
      }
      if (nz_idx[0] > nz_idx[1]) std::swap(nz_idx[0], nz_idx[1]);

      A_packed[m * PACKED_K + g * 2 + 0] = A_sparse[m * K + g * K_GROUP + nz_idx[0]];
      A_packed[m * PACKED_K + g * 2 + 1] = A_sparse[m * K + g * K_GROUP + nz_idx[1]];

      meta_uncompressed[m * PACKED_K + g * 2 + 0] = static_cast<uint32_t>(nz_idx[0]);
      meta_uncompressed[m * PACKED_K + g * 2 + 1] = static_cast<uint32_t>(nz_idx[1]);
    }
  }
  return true;
}

// Our own encoder: sparse A -> packed A + CUTLASS metadata buffer
static bool encode_sparse_to_cutlass_meta(
    const std::vector<__half> &A_sparse,
    std::vector<__half> &A_packed,
    std::vector<uint32_t> &meta_encoded) {
  std::vector<uint32_t> meta_uncompressed;
  if (!encode_sparse_to_packed_and_meta(A_sparse, A_packed, meta_uncompressed)) {
    return false;
  }
  encode_cutlass_metadata_interleaved2(meta_uncompressed, meta_encoded);
  return true;
}

int main() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Build a dense A
  std::vector<__half> A_dense(M * K);
  for (int i = 0; i < M * K; ++i) {
    A_dense[i] = __float2half(dist(rng));
  }

  // Offline pruning to 2:4 sparse A
  std::vector<__half> A_sparse;
  prune_dense_to_sparse_2of4(A_dense, A_sparse);

  // Encode using sparse->packed+metadata encoder (online)
  std::vector<__half> A_packed;
  std::vector<uint32_t> meta_uncompressed;
  bool ok_sparse = encode_sparse_to_packed_and_meta(A_sparse, A_packed, meta_uncompressed);

  // Encode metadata into CUTLASS layout
  std::vector<uint32_t> meta_encoded;
  encode_cutlass_metadata_interleaved2(meta_uncompressed, meta_encoded);

  // Decode CUTLASS metadata back to uncompressed indices
  std::vector<uint32_t> meta_decoded;
  decode_cutlass_metadata_interleaved2(meta_encoded, meta_decoded);

  bool ok_meta = (meta_decoded == meta_uncompressed);
  std::printf("CUTLASS metadata host encode/decode: %s\n", ok_meta ? "PASS" : "FAIL");

  // Compare our sparse encoder vs CUTLASS buffer produced by our wrapper
  std::vector<__half> A_packed2;
  std::vector<uint32_t> meta_encoded2;
  bool ok_sparse2 = encode_sparse_to_cutlass_meta(A_sparse, A_packed2, meta_encoded2);

  bool ok_packed = ok_sparse2 && (A_packed2.size() == A_packed.size());
  if (ok_packed) {
    for (size_t i = 0; i < A_packed.size(); ++i) {
      if (__half2float(A_packed2[i]) != __half2float(A_packed[i])) {
        ok_packed = false;
        break;
      }
    }
  }

  bool ok_meta_encoded = ok_sparse2 && (meta_encoded2 == meta_encoded);
  std::printf("Our encoder vs CUTLASS metadata buffer: %s\n", ok_meta_encoded ? "PASS" : "FAIL");
  std::printf("Our encoder packed A consistency: %s\n", ok_packed ? "PASS" : "FAIL");

  if (!ok_meta) {
    for (int i = 0; i < M * PACKED_K; ++i) {
      if (meta_uncompressed[i] != meta_decoded[i]) {
        std::printf("Mismatch at %d: orig=%u decoded=%u\n", i, meta_uncompressed[i], meta_decoded[i]);
        break;
      }
    }
  }

  // Print two rows as example
  for (int m = 0; m < 2; ++m) {
    std::printf("Row %d meta (0..15): ", m);
    for (int i = 0; i < PACKED_K; ++i) {
      std::printf("%u ", meta_uncompressed[m * PACKED_K + i]);
    }
    std::printf("\n");
  }

  return (ok_meta && ok_sparse && ok_sparse2 && ok_meta_encoded && ok_packed) ? EXIT_SUCCESS : EXIT_FAILURE;
}
