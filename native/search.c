// SIMD-accelerated IVF search for the fraud-detection backend.
//
// Compiled in the docker builder stage to a shared library that bun:ffi loads.
// Provides two entry points:
//
//   void  search_init(int16_t *vectors, uint8_t *labels, float *centroids,
//                     uint32_t *offsets, float *radii, float *decode_factor,
//                     int32_t n, int32_t k, int32_t d, int32_t nprobe);
//
//   int32_t search_query(const float *q);   // returns fraud count [0..5]
//
// The runtime owns the typed-array memory; we hold raw pointers and assume
// the buffers outlive the FFI session (true for our boot-and-serve lifecycle).
//
// Architectures: native NEON on aarch64, AVX2+FMA on x86_64.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#if defined(__aarch64__)
  #include <arm_neon.h>
  #define ARCH_NEON 1
#elif defined(__x86_64__) || defined(__amd64__)
  #include <immintrin.h>
  #define ARCH_AVX 1
#endif

#define D 14
#define MAX_NPROBE 256

typedef struct {
    float     decode_factor[16];   // padded so SIMD loads of 4 don't read past D
    int16_t  *vectors;             // n × D, row-major, cluster-sorted
    uint8_t  *labels;              // ceil(n/8), LSB-first packed
    float    *centroids;           // k × D, row-major
    uint32_t *offsets;             // k + 1
    float    *radii;               // k floats — sqrt(max squared L2 from centroid to any cluster member)
    int32_t   n, k, nprobe;
} ctx_t;

static ctx_t G;

void search_init(
    int16_t *vectors, uint8_t *labels, float *centroids,
    uint32_t *offsets, float *radii, float *decode_factor,
    int32_t n, int32_t k, int32_t d, int32_t nprobe
) {
    (void)d; // always 14
    G.vectors = vectors;
    G.labels = labels;
    G.centroids = centroids;
    G.offsets = offsets;
    G.radii = radii;
    memset(G.decode_factor, 0, sizeof(G.decode_factor));
    memcpy(G.decode_factor, decode_factor, sizeof(float) * D);
    G.n = n;
    G.k = k;
    G.nprobe = nprobe > MAX_NPROBE ? MAX_NPROBE : nprobe;
}

// ---- Squared L2 distance between query (float) and one centroid (float) ----

static inline float l2sq_centroid(const float *q, int32_t c) {
    const float *cv = G.centroids + (size_t)c * D;
#if ARCH_NEON
    // dims 0..3
    float32x4_t a0 = vsubq_f32(vld1q_f32(q),     vld1q_f32(cv));
    float32x4_t s  = vmulq_f32(a0, a0);
    // dims 4..7
    float32x4_t a1 = vsubq_f32(vld1q_f32(q + 4), vld1q_f32(cv + 4));
    s = vmlaq_f32(s, a1, a1);
    // dims 8..11
    float32x4_t a2 = vsubq_f32(vld1q_f32(q + 8), vld1q_f32(cv + 8));
    s = vmlaq_f32(s, a2, a2);
    float total = vaddvq_f32(s);
    // tail: dims 12, 13 (scalar)
    float t1 = q[12] - cv[12];
    float t2 = q[13] - cv[13];
    return total + t1 * t1 + t2 * t2;
#elif ARCH_AVX
    // dims 0..7
    __m256 q0 = _mm256_loadu_ps(q);
    __m256 c0 = _mm256_loadu_ps(cv);
    __m256 a0 = _mm256_sub_ps(q0, c0);
    __m256 s  = _mm256_mul_ps(a0, a0);
    // dims 8..13 via 128-wide load + scalar tail (avoid overrun)
    __m128 q1 = _mm_loadu_ps(q + 8);
    __m128 c1 = _mm_loadu_ps(cv + 8);
    __m128 a1 = _mm_sub_ps(q1, c1);
    __m128 s1 = _mm_mul_ps(a1, a1);
    // hsum s (8-wide) + s1 (4-wide) + tail
    float buf[8] __attribute__((aligned(32)));
    _mm256_storeu_ps(buf, s);
    float total = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    float buf1[4] __attribute__((aligned(16)));
    _mm_storeu_ps(buf1, s1);
    total += buf1[0]+buf1[1]+buf1[2]+buf1[3];
    float t1 = q[12] - cv[12];
    float t2 = q[13] - cv[13];
    return total + t1 * t1 + t2 * t2;
#else
    float s = 0;
    for (int d = 0; d < D; d++) {
        float diff = q[d] - cv[d];
        s += diff * diff;
    }
    return s;
#endif
}

// ---- Squared L2 distance between query (float) and one vector (int16) ----

static inline float l2sq_int16(const float *q, int32_t i) {
    const int16_t *v = G.vectors + (size_t)i * D;
#if ARCH_NEON
    // Load 8 int16 → 8 float32 → multiply by decode_factor → subtract → square → accumulate.
    // dims 0..3
    int16x4_t v0 = vld1_s16(v);
    float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(v0));
    f0 = vmulq_f32(f0, vld1q_f32(G.decode_factor));
    float32x4_t a0 = vsubq_f32(vld1q_f32(q), f0);
    float32x4_t s  = vmulq_f32(a0, a0);
    // dims 4..7
    int16x4_t v1 = vld1_s16(v + 4);
    float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(v1));
    f1 = vmulq_f32(f1, vld1q_f32(G.decode_factor + 4));
    float32x4_t a1 = vsubq_f32(vld1q_f32(q + 4), f1);
    s = vmlaq_f32(s, a1, a1);
    // dims 8..11
    int16x4_t v2 = vld1_s16(v + 8);
    float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(v2));
    f2 = vmulq_f32(f2, vld1q_f32(G.decode_factor + 8));
    float32x4_t a2 = vsubq_f32(vld1q_f32(q + 8), f2);
    s = vmlaq_f32(s, a2, a2);
    float total = vaddvq_f32(s);
    // tail: dims 12, 13
    float dv12 = (float)v[12] * G.decode_factor[12];
    float dv13 = (float)v[13] * G.decode_factor[13];
    float t1 = q[12] - dv12;
    float t2 = q[13] - dv13;
    return total + t1 * t1 + t2 * t2;
#elif ARCH_AVX
    // Load 8 int16 → 8 int32 → 8 float32 → multiply by decode_factor → ...
    __m128i v0 = _mm_loadu_si128((const __m128i *)(v));      // dims 0..7
    __m256i v032 = _mm256_cvtepi16_epi32(v0);
    __m256 f0 = _mm256_cvtepi32_ps(v032);
    __m256 df0 = _mm256_loadu_ps(G.decode_factor);
    f0 = _mm256_mul_ps(f0, df0);
    __m256 q0 = _mm256_loadu_ps(q);
    __m256 a0 = _mm256_sub_ps(q0, f0);
    __m256 s = _mm256_mul_ps(a0, a0);
    // dims 8..11 via 128-wide
    __m128i v1l = _mm_loadl_epi64((const __m128i *)(v + 8)); // 4 int16 in low 64 bits
    __m128i v1_32 = _mm_cvtepi16_epi32(v1l);                 // 4 int32
    __m128 f1 = _mm_cvtepi32_ps(v1_32);
    __m128 df1 = _mm_loadu_ps(G.decode_factor + 8);
    f1 = _mm_mul_ps(f1, df1);
    __m128 q1 = _mm_loadu_ps(q + 8);
    __m128 a1 = _mm_sub_ps(q1, f1);
    __m128 s1 = _mm_mul_ps(a1, a1);
    // hsum
    float buf[8] __attribute__((aligned(32)));
    _mm256_storeu_ps(buf, s);
    float total = buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6]+buf[7];
    float buf1[4] __attribute__((aligned(16)));
    _mm_storeu_ps(buf1, s1);
    total += buf1[0]+buf1[1]+buf1[2]+buf1[3];
    float dv12 = (float)v[12] * G.decode_factor[12];
    float dv13 = (float)v[13] * G.decode_factor[13];
    float t1 = q[12] - dv12;
    float t2 = q[13] - dv13;
    return total + t1 * t1 + t2 * t2;
#else
    float s = 0;
    for (int d = 0; d < D; d++) {
        float dv = (float)v[d] * G.decode_factor[d];
        float diff = q[d] - dv;
        s += diff * diff;
    }
    return s;
#endif
}

// ---- Bounded insertion-sort top-K ----

static inline void topk_init(float *dist, int32_t *idx, int n) {
    for (int t = 0; t < n; t++) { dist[t] = 1e30f; idx[t] = -1; }
}

static inline void topk_consider(
    float *dist, int32_t *idx, int n, float d, int32_t i
) {
    if (d >= dist[n - 1]) return;
    int j = n - 1;
    while (j > 0 && dist[j - 1] > d) {
        dist[j] = dist[j - 1];
        idx[j]  = idx[j - 1];
        j--;
    }
    dist[j] = d;
    idx[j]  = i;
}

// ---- Public hot path ----

int32_t search_query(const float *q) {
    float    probe_d[MAX_NPROBE];
    int32_t  probe_i[MAX_NPROBE];
    float    top5_d[5];
    int32_t  top5_i[5];

    int nprobe = G.nprobe;
    topk_init(probe_d, probe_i, nprobe);
    int32_t k = G.k;
    for (int32_t c = 0; c < k; c++) {
        float d = l2sq_centroid(q, c);
        topk_consider(probe_d, probe_i, nprobe, d, c);
    }

    topk_init(top5_d, top5_i, 5);
    // Cluster scan with triangle-inequality pruning.
    //
    // For cluster c with centroid distance d_c (squared) and radius r_c:
    //   any vector v in cluster c has dist(q, v) >= |sqrt(d_c) - r_c|
    // So squared L2 lower bound is `(sqrt(d_c) - r_c)^2` when sqrt(d_c) > r_c.
    // If that lower bound > top5_d[4], no vector in c can enter the top-5 →
    // skip the entire cluster scan.
    for (int p = 0; p < nprobe; p++) {
        int32_t c = probe_i[p];
        if (c < 0) continue;
        const float dc = probe_d[p];                 // squared centroid distance
        const float rc = G.radii[c];
        const float sqrt_dc = sqrtf(dc);
        if (sqrt_dc > rc) {
            const float diff = sqrt_dc - rc;
            const float lower_bound_sq = diff * diff;
            if (lower_bound_sq >= top5_d[4]) continue;  // safe prune
        }
        uint32_t lo = G.offsets[c];
        uint32_t hi = G.offsets[c + 1];
        for (uint32_t i = lo; i < hi; i++) {
            float d = l2sq_int16(q, (int32_t)i);
            topk_consider(top5_d, top5_i, 5, d, (int32_t)i);
        }
    }

    int32_t frauds = 0;
    const uint8_t *labels = G.labels;
    for (int t = 0; t < 5; t++) {
        int32_t i = top5_i[t];
        if (i < 0) continue;
        frauds += (labels[i >> 3] >> (i & 7)) & 1;
    }
    return frauds;
}
