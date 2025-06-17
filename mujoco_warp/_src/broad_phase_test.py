# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Any

import numpy as np
import warp as wp















# Collision filtering
@wp.func
def proceed_broad_phase(group_a: int, group_b: int) -> bool:
  if group_a == 0 or group_b == 0:
    return False
  if group_a > 0:
    return group_a == group_b or group_b < 0
  if group_a < 0:
    return group_a != group_b


@wp.func
def check_aabb_overlap(
  box1_lower: wp.vec3, box1_upper: wp.vec3, box1_cutoff: float, box2_lower: wp.vec3, box2_upper: wp.vec3, box2_cutoff: float
) -> bool:
  cutoff_combined = max(box1_cutoff, box2_cutoff)
  return (
    box1_lower[0] <= box2_upper[0] + cutoff_combined
    and box1_upper[0] >= box2_lower[0] - cutoff_combined
    and box1_lower[1] <= box2_upper[1] + cutoff_combined
    and box1_upper[1] >= box2_lower[1] - cutoff_combined
    and box1_lower[2] <= box2_upper[2] + cutoff_combined
    and box1_upper[2] >= box2_lower[2] - cutoff_combined
  )


@wp.func
def write_pair(
  pair: wp.vec2i,
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):
  pairid = wp.atomic_add(num_candidate_pair, 0, 1)

  if pairid >= max_candidate_pair:
    return

  candidate_pair[pairid] = pair


@wp.func
def nxn_broadphase_precomputed_pairs(
  # Input arrays
  elementid: int,
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),  
  geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
  nxn_geom_pair: wp.array(
    dtype=wp.vec2i, ndim=1
  ),  # Precompute, all pairs that need to be checked - static information, not associated with a specific environment - make optional, empty list (None) allowed
  # Output arrays
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):
  pair = nxn_geom_pair[elementid]
  geom1 = pair[0]
  geom2 = pair[1]

  if check_aabb_overlap(
    geom_bounding_box_lower[geom1],
    geom_bounding_box_upper[geom1],
    geom_cutoff[geom1],
    geom_bounding_box_lower[geom2],
    geom_bounding_box_upper[geom2],
    geom_cutoff[geom2],
  ):
    write_pair(
      pair,
      candidate_pair,
      num_candidate_pair,
      max_candidate_pair,
    )


@wp.func
def isqrt(y: int) -> int:
    L = int(0)
    R = int(y + 1)
    while L != R - 1:
        M = (L + R) // 2
        if M * M <= y:
            L = M
        else:
            R = M
    return L


@wp.func
def nxn_broadphase(
  # Input arrays
  elementid: int,
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  num_boxes : int,
  geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
  collision_group: wp.array(dtype=int, ndim=1),  # per-geom
  # Output arrays
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):
  # For lower triangle (i > j), no diagonal, elementid in [0, n*(n-1)//2)
  i = (isqrt(8 * elementid + 1) + 1) // 2
  j = elementid - i * (i - 1) // 2
  pair = wp.vec2i(j, i)  # with i > j guaranteed

  if i >= num_boxes or j >= num_boxes:
    wp.printf("%d geom1=%d, geom2=%d\n",elementid, j, i)
    return

  geom1 = pair[0]
  geom2 = pair[1]

  if collision_group.shape[0] > 0 and not proceed_broad_phase(collision_group[geom1], collision_group[geom2]):
    return

  # wp.printf("geom1=%d, geom2=%d\n", geom1, geom2)

  if check_aabb_overlap(
    geom_bounding_box_lower[geom1],
    geom_bounding_box_upper[geom1],
    geom_cutoff[geom1],
    geom_bounding_box_lower[geom2],
    geom_bounding_box_upper[geom2],
    geom_cutoff[geom2],
  ):
    write_pair(
      pair,
      candidate_pair,
      num_candidate_pair,
      max_candidate_pair,
    )



















wp.set_module_options({"enable_backward": False})


@wp.kernel
def nxn_broadphase_kernel(
  # Input arrays
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  num_boxes : int,
  geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
  collision_group: wp.array(dtype=int, ndim=1),  # per-geom
  # Output arrays
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):
  elementid = wp.tid()
  nxn_broadphase(
    elementid,
    geom_bounding_box_lower,
    geom_bounding_box_upper,
    num_boxes,
    geom_cutoff,
    collision_group,
    candidate_pair,
    num_candidate_pair,
    max_candidate_pair,
  )


def find_overlapping_pairs_np(box_lower: np.ndarray, box_upper: np.ndarray):
    """
    Brute-force n^2 algorithm to find all overlapping bounding box pairs.
    Each box is axis-aligned, defined by min (lower) and max (upper) corners.
    Returns a list of (i, j) pairs with i < j, where boxes i and j overlap.
    """
    n = box_lower.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # Check for overlap in all three axes
            if (
                (box_lower[i, 0] <= box_upper[j, 0] and box_upper[i, 0] >= box_lower[j, 0]) and
                (box_lower[i, 1] <= box_upper[j, 1] and box_upper[i, 1] >= box_lower[j, 1]) and
                (box_lower[i, 2] <= box_upper[j, 2] and box_upper[i, 2] >= box_lower[j, 2])
            ):
                pairs.append((i, j))
    return pairs


def test_nxn_broadphase():

  # Create random bounding boxes in min-max format
  ngeom = 200  # You can parameterize this as needed
  # Generate random centers and sizes
  centers = np.random.rand(ngeom, 3) * 10.0
  sizes = np.random.rand(ngeom, 3) * 2.0  # box half-extent up to 1.0 in each direction
  geom_bounding_box_lower = centers - sizes
  geom_bounding_box_upper = centers + sizes

  pairs_np = find_overlapping_pairs_np(geom_bounding_box_lower, geom_bounding_box_upper)
  # print(pairs_np)

  

  # The number of elements in the lower triangular part of an n x n matrix (excluding the diagonal)
  # is given by n * (n - 1) // 2
  num_lower_tri_elements = ngeom * (ngeom - 1) // 2

  geom_bounding_box_lower_wp = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
  geom_bounding_box_upper_wp = wp.array(geom_bounding_box_upper, dtype=wp.vec3)
  geom_cutoff = wp.array(np.zeros(ngeom, dtype=np.float32))
  collision_group = wp.array(np.ones(ngeom, dtype=np.int32)) 
  num_candidate_pair = wp.array([0, ], dtype=wp.int32)
  max_candidate_pair = num_lower_tri_elements
  candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype= wp.vec2i)

  wp.launch(
    nxn_broadphase_kernel,
    dim=num_lower_tri_elements,
    inputs=[geom_bounding_box_lower_wp, geom_bounding_box_upper_wp, ngeom, geom_cutoff, collision_group],
    outputs=[candidate_pair, num_candidate_pair, max_candidate_pair],
  )
  wp.synchronize()

  pairs_wp = candidate_pair.numpy()
  num_candidate_pair = num_candidate_pair.numpy()[0]

  if len(pairs_np) != num_candidate_pair:
    print(f"len(pairs_np)={len(pairs_np)}, num_candidate_pair={num_candidate_pair}")
    # print("pairs_np:", pairs_np)
    # print("pairs_wp[:num_candidate_pair]:", pairs_wp[:num_candidate_pair])
    assert len(pairs_np) == num_candidate_pair

  # Ensure every element in pairs_wp is also present in pairs_np
  pairs_np_set = set(tuple(pair) for pair in pairs_np)
  for pair in pairs_wp[:num_candidate_pair]:
    assert tuple(pair) in pairs_np_set, f"Pair {tuple(pair)} from Warp not found in numpy pairs"

  print(len(pairs_np))



def test_sap_broadphase():
  pass


if __name__ == "__main__":
  wp.clear_kernel_cache()
  test_nxn_broadphase()
  test_sap_broadphase()
