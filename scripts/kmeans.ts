// Mini-batch k-means used by the build-time preprocessor (spec §3.2 step 6).
// Operates on dequantized float vectors (not int8). All helpers are exported
// so the test suite can exercise them in isolation.

// --- Seeded random ------------------------------------------------------------

// Tiny linear-congruential generator (Numerical Recipes constants). Returns a
// stateful function that yields uniformly-distributed floats in [0, 1).
export function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

// --- Nearest centroid ---------------------------------------------------------

// Find the index of the centroid (in `centroids`, shape (k, d)) minimizing
// squared L2 distance from `point[pointOffset .. pointOffset+d-1]`.
export function nearestCentroidIndex(
  point: Float32Array,
  pointOffset: number,
  centroids: Float32Array,
  k: number,
  d: number,
): number {
  let bestIdx = 0;
  let bestDist = Number.POSITIVE_INFINITY;
  for (let c = 0; c < k; c++) {
    const cBase = c * d;
    let sum = 0;
    for (let dim = 0; dim < d; dim++) {
      const diff = (point[pointOffset + dim] as number) - (centroids[cBase + dim] as number);
      sum += diff * diff;
    }
    if (sum < bestDist) {
      bestDist = sum;
      bestIdx = c;
    }
  }
  return bestIdx;
}

// Squared L2 distance from `point[offset..offset+d]` to `centroids[c*d..(c+1)*d]`.
function squaredDistanceToCentroid(
  point: Float32Array,
  offset: number,
  centroids: Float32Array,
  c: number,
  d: number,
): number {
  const cBase = c * d;
  let sum = 0;
  for (let dim = 0; dim < d; dim++) {
    const diff = (point[offset + dim] as number) - (centroids[cBase + dim] as number);
    sum += diff * diff;
  }
  return sum;
}

// --- k-means++ seeding --------------------------------------------------------

// Allocates `centroids = Float32Array(k*d)` and fills with the chosen seeds.
// Returns both the centroid buffer and the indices of the chosen rows.
export function kmeansPlusPlusSeed(
  flat: Float32Array,
  n: number,
  d: number,
  k: number,
  rng: () => number,
): { centroids: Float32Array; chosenIndices: Uint32Array } {
  const centroids = new Float32Array(k * d);
  const chosen = new Uint32Array(k);

  // First centroid: uniformly random.
  let firstIdx = Math.floor(rng() * n);
  if (firstIdx >= n) firstIdx = n - 1;
  chosen[0] = firstIdx;
  for (let dim = 0; dim < d; dim++) {
    centroids[dim] = flat[firstIdx * d + dim] as number;
  }

  // For each subsequent centroid, sample proportional to D(x)^2.
  const distSq = new Float64Array(n); // reused across outer iterations
  for (let i = 0; i < n; i++) {
    distSq[i] = squaredDistanceToCentroid(flat, i * d, centroids, 0, d);
  }

  for (let c = 1; c < k; c++) {
    // Cumulative sum, sample by inverse CDF.
    let total = 0;
    for (let i = 0; i < n; i++) total += distSq[i] as number;
    let pickedIdx = n - 1;
    if (total <= 0) {
      // All points coincide with existing centroids — fall back to uniform.
      pickedIdx = Math.floor(rng() * n);
      if (pickedIdx >= n) pickedIdx = n - 1;
    } else {
      const target = rng() * total;
      let acc = 0;
      for (let i = 0; i < n; i++) {
        acc += distSq[i] as number;
        if (acc >= target) {
          pickedIdx = i;
          break;
        }
      }
    }
    chosen[c] = pickedIdx;
    for (let dim = 0; dim < d; dim++) {
      centroids[c * d + dim] = flat[pickedIdx * d + dim] as number;
    }

    // Update distSq with min(distSq[i], dist(point_i, new_centroid))
    for (let i = 0; i < n; i++) {
      const dNew = squaredDistanceToCentroid(flat, i * d, centroids, c, d);
      if (dNew < (distSq[i] as number)) distSq[i] = dNew;
    }
  }

  return { centroids, chosenIndices: chosen };
}

// --- Mini-batch update --------------------------------------------------------

// Apply a single mini-batch step. For each sampled point, find its nearest
// centroid, increment that centroid's `counts`, then nudge the centroid toward
// the point with learning rate `1 / counts[c]` (Bottou & Bengio 1995). Updates
// `centroids` and `counts` in place; no allocation.
export function miniBatchUpdate(
  centroids: Float32Array,
  counts: Float32Array,
  flat: Float32Array,
  sampleIndices: Uint32Array,
  d: number,
  k: number,
): void {
  const m = sampleIndices.length;
  for (let s = 0; s < m; s++) {
    const idx = sampleIndices[s] as number;
    const c = nearestCentroidIndex(flat, idx * d, centroids, k, d);
    counts[c] = (counts[c] as number) + 1;
    const lr = 1 / (counts[c] as number);
    const cBase = c * d;
    const pBase = idx * d;
    for (let dim = 0; dim < d; dim++) {
      const cv = centroids[cBase + dim] as number;
      const pv = flat[pBase + dim] as number;
      centroids[cBase + dim] = (1 - lr) * cv + lr * pv;
    }
  }
}

// --- miniBatchKMeans ----------------------------------------------------------

// Run k-means++ seeding then `iters` mini-batch updates of size `batch`.
// Returns float-space centroids and a final assignment for every point.
export function miniBatchKMeans(
  flat: Float32Array,
  n: number,
  d: number,
  k: number,
  iters: number,
  batch: number,
  seed: number,
): { centroids: Float32Array; assignments: Uint32Array } {
  const rng = lcg(seed);
  const { centroids } = kmeansPlusPlusSeed(flat, n, d, k, rng);
  const counts = new Float32Array(k);
  const sampleIndices = new Uint32Array(batch);

  for (let it = 0; it < iters; it++) {
    for (let s = 0; s < batch; s++) {
      let idx = Math.floor(rng() * n);
      if (idx >= n) idx = n - 1;
      sampleIndices[s] = idx;
    }
    miniBatchUpdate(centroids, counts, flat, sampleIndices, d, k);
  }

  // Final hard assignment over every point.
  const assignments = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    assignments[i] = nearestCentroidIndex(flat, i * d, centroids, k, d);
  }
  return { centroids, assignments };
}
