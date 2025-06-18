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



wp.set_module_options({"enable_backward": False})


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


# Based on binary search
@wp.func
def get_lower_triangular_indices(index: int, matrix_size: int) -> wp.vec2i:
  total = (matrix_size * (matrix_size - 1)) >> 1
  if index >= total:
    # In Warp, we can't throw, so return an invalid pair
    return wp.vec2i(-1, -1)

  low = int(0)
  high = matrix_size - 1
  while low < high:
    mid = (low + high) >> 1
    count = (mid * (2 * matrix_size - mid - 1)) >> 1
    if count <= index:
      low = mid + 1
    else:
      high = mid
  r = low - 1
  f = (r * (2 * matrix_size - r - 1)) >> 1
  c = (index - f) + r + 1
  return wp.vec2i(r, c)


@wp.func
def nxn_broadphase(
  # Input arrays
  elementid: int,
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  first_box_index : int,
  num_boxes: int,
  geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
  collision_group: wp.array(dtype=int, ndim=1),  # per-geom
  # Output arrays
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):
  pair = get_lower_triangular_indices(elementid, num_boxes)
  pair[0] += first_box_index
  pair[1] += first_box_index

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


@wp.func
def _binary_search(values: wp.array(dtype=Any), value: Any, lower: int, upper: int) -> int:
  while lower < upper:
    mid = (lower + upper) >> 1
    if values[mid] > value:
      upper = mid
    else:
      lower = mid + 1

  return upper






# Determines the amount of geoms that need to be checked for overlap
# Can vary a lot between different threads
# Use load balancing (by using a scan) based on the result of sap_range
@wp.func
def sap_range(
  elementid: int,
  ngeom: int,
  sap_projection_lower_in: wp.array(dtype=wp.float, ndim=1),
  sap_projection_upper_in: wp.array(dtype=wp.float, ndim=1),
  sap_sort_index_in: wp.array(dtype=wp.int, ndim=1),
):
  # current bounding geom
  idx = sap_sort_index_in[elementid]

  upper = sap_projection_upper_in[idx]

  limit = _binary_search(sap_projection_lower_in, upper, elementid + 1, ngeom)
  limit = wp.min(ngeom - 1, limit)

  # range of geoms for the sweep and prune process
  return limit - elementid



@wp.func
def process_single_sap_pair(
  pair: wp.vec2i,
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  geom_cutoff: wp.array(dtype=wp.float, ndim=1),  # per-geom (take the max)
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):
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

# Uses load balancing over envs
@wp.func
def sap_find_overlaps(
  worldgeomid: int,

):  
  nworldgeom = nworld_in * ngeom
  nworkpackages = sap_cumulative_sum_in[nworldgeom - 1]

  while worldgeomid < nworkpackages:
    # binary search to find current and next geom pair indices
    i = _binary_search(sap_cumulative_sum_in, worldgeomid, 0, nworldgeom)
    j = i + worldgeomid + 1

    if i > 0:
      j -= sap_cumulative_sum_in[i - 1]

    worldid = i // ngeom
    i = i % ngeom
    j = j % ngeom

    # get geom indices and swap if necessary
    geom1 = sap_sort_index_in[worldid, i]
    geom2 = sap_sort_index_in[worldid, j]

    # find linear index of (geom1, geom2) in upper triangular nxn_pairid
    if geom2 < geom1:
      idx = upper_tri_index(ngeom, geom2, geom1)
    else:
      idx = upper_tri_index(ngeom, geom1, geom2)

    if nxn_pairid[idx] < -1:
      worldgeomid += nsweep_in
      continue

    pair = wp.vec2i(geom1, geom2)
    process_single_sap_pair(
      pair,
      geom_bounding_box_lower,
      geom_bounding_box_upper,
      geom_cutoff,
      candidate_pair,
    )

    worldgeomid += nsweep_in


# @wp.func
# def broad_phase_aabb(
#   geom_bounding_box_lower: wp.array(dtype=vec3, ndim=1),
#   geom_bounding_box_upper: wp.array(dtype=vec3, ndim=1),
#   geom_cutoff: wp.array(dtype=wp.float, ndim=1),  # per-geom (take the max)
#   collision_group: wp.array(
#     dtype=int, ndim=1
#   ),  # per-geom collision group. If number is negative, it collides with all the positive collision groups, and collides with all negative groups except itself. If number is positive, the the geom only collides with geoms that have the same collision group, and all negative groups. If number is zero, the geom collides with nothing.
#   nxn_geom_pair: wp.array(dtype=vec2i, ndim=1)
#   | None,  # Precompute, all pairs that need to be checkd - static information, not associated with a specific environment - make optional, empty list (None) allowed
#   # outputs
#   candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
#   num_candidate_pair: wp.array(dtype=wp.int, ndim=1),  # Size one array
#   max_candidate_pair: int,
# ):
#   pass



@wp.kernel
def sap_broadphase_kernel(
  # Input arrays
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  num_boxes: int,
  geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
  collision_group: wp.array(dtype=int, ndim=1),  # per-geom
  # Output arrays
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):


  pass



@wp.func
def sap_project_aabb(
  elementid: int,
  direction: wp.vec3, # Must be normalized
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  geom_cutoff: wp.array(dtype=wp.float, ndim=1),  # per-geom (take the max)
) -> wp.vec2:
  lower = geom_bounding_box_lower[elementid]
  upper = geom_bounding_box_upper[elementid]
  cutoff = geom_cutoff[elementid]

  half_size = 0.5 * (upper - lower)
  half_size = wp.vec3(half_size[0]+ cutoff, half_size[1]+ cutoff, half_size[2]+ cutoff)
  radius = wp.dot(direction, half_size)
  center = wp.dot(direction, 0.5 * (lower + upper))
  return wp.vec2(center - radius, center + radius)


@wp.kernel
def sap_project_aabb_kernel(
  direction: wp.vec3, # Must be normalized
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  geom_cutoff: wp.array(dtype=wp.float, ndim=1),  # per-geom (take the max)
):
  elementid = wp.tid()
  sap_project_aabb(
    elementid,
    direction,
    geom_bounding_box_lower,
    geom_bounding_box_upper,
    geom_cutoff,
  )





# Currently all sections that get sorted must have the same length
# TODO: Add support for variable lengths
def create_sap_sort_func(sort_length: int):
  @wp.func
  def sap_sort_index(
    startid: int,
    sap_projection_lower_in: wp.array(dtype=float),
    sap_sort_index_in: wp.array(dtype=int),
  ): 
    # Load input into shared memory
    keys = wp.tile_load(sap_projection_lower_in, offset=startid, shape=sort_length, storage="shared")
    values = wp.tile_load(sap_sort_index_in, offset=startid, shape=sort_length, storage="shared")

    # Perform in-place sorting
    wp.tile_sort(keys, values)

    # Store sorted shared memory into output arrays
    wp.tile_store(sap_projection_lower_in, keys, offset=startid)
    wp.tile_store(sap_sort_index_in, values, offset=startid)

  return sap_sort_index


def create_sap_sort_kernel(sort_length: int):
  @wp.kernel
  def sap_sort_index_kernel(
    startid: int,
    sap_projection_lower_in: wp.array(dtype=float),
    sap_sort_index_in: wp.array(dtype=int),
  ):
    sap_sort_index(startid, sap_projection_lower_in, sap_sort_index_in)
  return sap_sort_index_kernel



def sap_broadphase(
    geom_bounding_box_lower_wp: wp.array(dtype=wp.vec3, ndim=1),
    geom_bounding_box_upper_wp: wp.array(dtype=wp.vec3, ndim=1),
    ngeom: int,
    geom_cutoff: wp.array(dtype=float, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,
):

  # TODO: direction

  # random fixed direction
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)

  wp.launch(
    kernel=sap_project_aabb_kernel,
    dim=ngeom,
    inputs=[
      direction,
      geom_bounding_box_lower_wp,
      geom_bounding_box_upper_wp,
      geom_cutoff,
    ]
  )

  # First: sort lower projections - keep track of the original index -> track array
  # Then Extract the location of all objects with -1 as collision group -> index array
  # Run collision detection only for objects with collision group -1 - use load balancing

  # Second: Sort according to groups starting reusing the track array as payload
  # Now run collision detection for all objects with collision group > 0 - only collide them against the same group since -1 group was already handled


  pass






@wp.kernel
def nxn_broadphase_kernel(
  # Input arrays
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  num_boxes: int,
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
    0,
    num_boxes,
    geom_cutoff,
    collision_group,
    candidate_pair,
    num_candidate_pair,
    max_candidate_pair,
  )


@wp.kernel
def nxn_broadphase_kernel_identical_envs(
  # Input arrays
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  num_boxes: int,
  geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
  collision_group: wp.array(dtype=int, ndim=1),  # per-geom
  num_boxes_per_env: int,
  # Output arrays
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):
  thread_id = wp.tid()
  
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
        (box_lower[i, 0] <= box_upper[j, 0] and box_upper[i, 0] >= box_lower[j, 0])
        and (box_lower[i, 1] <= box_upper[j, 1] and box_upper[i, 1] >= box_lower[j, 1])
        and (box_lower[i, 2] <= box_upper[j, 2] and box_upper[i, 2] >= box_lower[j, 2])
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
  num_candidate_pair = wp.array(
    [
      0,
    ],
    dtype=wp.int32,
  )
  max_candidate_pair = num_lower_tri_elements
  candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

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
  num_candidate_pair = wp.array(
    [
      0,
    ],
    dtype=wp.int32,
  )
  max_candidate_pair = num_lower_tri_elements
  candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

  sap_broadphase(
    geom_bounding_box_lower_wp,
    geom_bounding_box_upper_wp,
    ngeom,
    geom_cutoff,
    collision_group,
    candidate_pair,
    num_candidate_pair,
    max_candidate_pair,
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


if __name__ == "__main__":
  wp.clear_kernel_cache()
  test_nxn_broadphase()
  test_sap_broadphase()
