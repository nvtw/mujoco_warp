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
  sap_projection_lower_in: wp.array(dtype=float, ndim=1),
  sap_projection_upper_in: wp.array(dtype=float, ndim=1),
  sap_sort_index_in: wp.array(dtype=int, ndim=1),
):
  # current bounding geom
  idx = sap_sort_index_in[elementid]

  upper = sap_projection_upper_in[idx]

  limit = _binary_search(sap_projection_lower_in, upper, elementid + 1, ngeom)
  limit = wp.min(ngeom - 1, limit)

  # range of geoms for the sweep and prune process
  return limit - elementid


@wp.kernel
def sap_range_kernel(
  ngeom: int,
  sap_projection_lower_in: wp.array(dtype=float, ndim=1),
  sap_projection_upper_in: wp.array(dtype=float, ndim=1),
  sap_sort_index_in: wp.array(dtype=int, ndim=1),
  sap_range_out: wp.array(dtype=int, ndim=1),
):
  elementid = wp.tid()
  if elementid >= ngeom:
    return
  count = sap_range(elementid, ngeom, sap_projection_lower_in, sap_projection_upper_in, sap_sort_index_in)
  sap_range_out[elementid] = count


@wp.kernel
def sap_range_indexed_kernel(
  indexer: wp.array(dtype=int, ndim=1),
  indexer_length : wp.array(dtype=int, ndim=1), # Length 1 array
  ngeom: int,
  sap_projection_lower_in: wp.array(dtype=float, ndim=1),
  sap_projection_upper_in: wp.array(dtype=float, ndim=1),
  sap_sort_index_in: wp.array(dtype=int, ndim=1),
  sap_range_out: wp.array(dtype=int, ndim=1),
):
  elementid = wp.tid()
  if elementid >= indexer_length[0]:
    return
  count = sap_range(indexer[elementid], ngeom, sap_projection_lower_in, sap_projection_upper_in, sap_sort_index_in)
  sap_range_out[elementid] = count


@wp.func
def process_single_sap_pair(
  pair: wp.vec2i,
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  geom_cutoff: wp.array(dtype=float, ndim=1), 
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


@wp.kernel
def sap_broadphase_kernel(
  # Input arrays
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  num_boxes: int,
  num_minus_one_boxes: wp.array(dtype=int, ndim=1),# Size one array
  sap_sort_index_in: wp.array(dtype=int, ndim=1),
  sap_cumulative_sum_in: wp.array(dtype=int, ndim=1),
  geom_cutoff: wp.array(dtype=float, ndim=1),
  nsweep_in: int,
  # Output arrays
  candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
  num_candidate_pair: wp.array(dtype=int, ndim=1),  # Size one array
  max_candidate_pair: int,
):
  geomid = wp.tid()

  if (num_boxes == -1):
    num_boxes = num_minus_one_boxes[0]
  else:
    num_boxes = num_boxes - num_minus_one_boxes[0]
 
  nworkpackages = int(0)
  if num_boxes > 0:
    nworkpackages = sap_cumulative_sum_in[num_boxes - 1]

  while geomid < nworkpackages:
    # binary search to find current and next geom pair indices
    i = _binary_search(sap_cumulative_sum_in, geomid, 0, num_boxes)
    j = i + geomid + 1

    if i > 0:
      j -= sap_cumulative_sum_in[i - 1]



    # get geom indices and swap if necessary
    geom1 = sap_sort_index_in[i]
    geom2 = sap_sort_index_in[j]

    if(geom1 > geom2):
      tmp = geom1
      geom1 = geom2
      geom2 = tmp

    process_single_sap_pair(
      wp.vec2i(geom1, geom2),
      geom_bounding_box_lower,
      geom_bounding_box_upper,
      geom_cutoff,
      candidate_pair,
      num_candidate_pair,
      max_candidate_pair,
    )

    geomid += nsweep_in



@wp.func
def sap_project_aabb(
  elementid: int,
  direction: wp.vec3, # Must be normalized
  geom_bounding_box_lower: wp.array(dtype=wp.vec3, ndim=1),
  geom_bounding_box_upper: wp.array(dtype=wp.vec3, ndim=1),
  geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
  collision_group: wp.array(dtype=int, ndim=1),
) -> wp.vec2:
  lower = geom_bounding_box_lower[elementid]
  upper = geom_bounding_box_upper[elementid]
  cutoff = geom_cutoff[elementid]
  group = collision_group[elementid]

  if(group == 0):
    # Collision group 0 does not collide with anything
    return wp.vec2(1000000.0 + float(elementid), 1000000.0 + float(elementid))

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
  geom_cutoff: wp.array(dtype=float, ndim=1),  # per-geom (take the max)
  collision_group: wp.array(dtype=int, ndim=1),
  sap_projection_lower_out: wp.array(dtype=float, ndim=1),
  sap_projection_upper_out: wp.array(dtype=float, ndim=1),
  sap_sort_index_out: wp.array(dtype=int, ndim=1),
):
  elementid = wp.tid()
  proj = sap_project_aabb(
    elementid,
    direction,
    geom_bounding_box_lower,
    geom_bounding_box_upper,
    geom_cutoff,
    collision_group
  )
  sap_projection_lower_out[elementid] = proj[0]
  sap_projection_upper_out[elementid] = proj[1]
  sap_sort_index_out[elementid] = elementid

@wp.kernel
def sap_assign_collision_group_kernel(
  collision_group: wp.array(dtype=int, ndim=1),
  sap_sort_index_out: wp.array(dtype=int, ndim=1),
  collision_group_tmp: wp.array(dtype=int, ndim=1),
  index_tracking_tmp: wp.array(dtype=int, ndim=1),
  index_tracking_tmp_indexer: wp.array(dtype=int, ndim=1),
):
  id = wp.tid()
  source_id = sap_sort_index_out[id]
  group = collision_group[source_id]
  collision_group_tmp[id] = group
  if group == -1:
    index = wp.atomic_add(index_tracking_tmp_indexer, 0, 1)
    index_tracking_tmp[index] = id


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


def sap_broadphase(
    geom_bounding_box_lower_wp: wp.array(dtype=wp.vec3, ndim=1),
    geom_bounding_box_upper_wp: wp.array(dtype=wp.vec3, ndim=1),
    num_boxes: int,
    geom_cutoff: wp.array(dtype=float, ndim=1),
    collision_group: wp.array(dtype=int, ndim=1),
    candidate_pair: wp.array(dtype=wp.vec2i, ndim=1),
    num_candidate_pair: wp.array(dtype=int, ndim=1),
    max_candidate_pair: int,

    # Temp memory
    sap_projection_lower: wp.array(dtype=float, ndim=1),
    sap_projection_upper: wp.array(dtype=float, ndim=1),
    sap_sort_index: wp.array(dtype=int, ndim=1),
    collision_group_tmp: wp.array(dtype=int, ndim=1),
    index_tracking_tmp: wp.array(dtype=int, ndim=1),
    index_tracking_tmp_counter: wp.array(dtype=int, ndim=1), # Length 1 array
    sap_range: wp.array(dtype=int, ndim=1),
    sap_cumulative_sum: wp.array(dtype=int, ndim=1),
):

  # TODO: direction

  # random fixed direction
  direction = wp.vec3(0.5935, 0.7790, 0.1235)
  direction = wp.normalize(direction)

  wp.launch(
    kernel=sap_project_aabb_kernel,
    dim=num_boxes,
    inputs=[
      direction,
      geom_bounding_box_lower_wp,
      geom_bounding_box_upper_wp,
      geom_cutoff,
      collision_group
    ],
    outputs=[
      sap_projection_lower,
      sap_projection_upper,
      sap_sort_index,
    ]
  )

  # First: sort lower projections - keep track of the original index
  # Then Extract the location of all objects with -1 as collision group 
  # Run collision detection only for objects with collision group -1 - use load balancing

  wp.utils.radix_sort_pairs(
    sap_projection_lower,
    sap_sort_index,
    num_boxes
    )
  

  wp.launch(
    kernel=sap_assign_collision_group_kernel,
    dim=num_boxes,
    inputs=[
      collision_group,
      sap_sort_index
    ],
    outputs=[
      collision_group_tmp,
      index_tracking_tmp,
      index_tracking_tmp_counter,
    ]
  )


  # Process collision group -1
  wp.launch(
    kernel=sap_range_indexed_kernel,
    dim=num_boxes,
    inputs=[
      index_tracking_tmp,
      index_tracking_tmp_counter,
      num_boxes,
      sap_projection_lower,
      sap_projection_upper,
      sap_sort_index,
      sap_range,
    ]
  )

  # TODO: Is it possible to only do the scan over index_tracking_tmp_indexer[0] elements?
  wp.utils.array_scan(sap_range.reshape(-1), sap_cumulative_sum, True)

  nsweep_in = 5 * num_boxes
  wp.launch(
    kernel=sap_broadphase_kernel,
    dim=nsweep_in,
    inputs=[
      geom_bounding_box_lower_wp,
      geom_bounding_box_upper_wp,
      -1,
      index_tracking_tmp_counter,
      sap_sort_index,
      sap_cumulative_sum,
      geom_cutoff,
      nsweep_in,
    ],
    outputs=[
      candidate_pair,
      num_candidate_pair,
      max_candidate_pair,
    ]
  )


  # wp.synchronize()
  # # print(collision_group_tmp.numpy())
  # # print(index_tracking_tmp.numpy())
  # print(sap_projection_lower_out.numpy())
  # print(sap_sort_index_out.numpy())
  # print(index_tracking_tmp_indexer.numpy())
  # print(num_candidate_pair.numpy())

  # Process collision groups > 0
  # Requires sort_pairs to be a stable sort
  wp.utils.radix_sort_pairs(
    collision_group_tmp,
    sap_sort_index,
    num_boxes
    )

  # TODO: Ensure that negative collision groups end up at the end of the array
  wp.launch(
    kernel=sap_range_kernel,
    dim=num_boxes,
    inputs=[
      num_boxes,
      sap_projection_lower,
      sap_projection_upper,
      sap_sort_index,
      sap_range,
    ]
  )

  wp.utils.array_scan(sap_range.reshape(-1), sap_cumulative_sum, True)

  # Second: Sort according to groups starting reusing the track array as payload
  # Now run collision detection for all objects with collision group > 0 - only collide them against the same group since -1 group was already handled

  wp.launch(
    kernel=sap_broadphase_kernel,
    dim=nsweep_in,
    inputs=[
      geom_bounding_box_lower_wp,
      geom_bounding_box_upper_wp,
      num_boxes,
      index_tracking_tmp_counter,
      sap_sort_index,
      sap_cumulative_sum,
      geom_cutoff,
      nsweep_in,
    ],
    outputs=[
      candidate_pair,
      num_candidate_pair,
      max_candidate_pair,
    ]
  )


  # wp.synchronize()
  # print("End ------------------------")
  # # print(index_tracking_tmp.numpy())
  # print(sap_projection_lower_out.numpy())
  # print(sap_sort_index_out.numpy())
  # print(sap_cumulative_sum.numpy())
  # print(index_tracking_tmp_indexer.numpy())
  # print(num_candidate_pair.numpy())
  


# Collision filtering
def proceed_broad_phase(group_a: int, group_b: int) -> bool:
  if group_a == 0 or group_b == 0:
    return False
  if group_a > 0:
    return group_a == group_b or group_b < 0
  if group_a < 0:
    return group_a != group_b


def find_overlapping_pairs_np(box_lower: np.ndarray, box_upper: np.ndarray, 
                              cutoff: np.ndarray, collision_group: np.ndarray):
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
      cutoff_combined = max(cutoff[i], cutoff[j])
      if not proceed_broad_phase(collision_group[i], collision_group[j]):
        continue

      if (
        (box_lower[i, 0] <= box_upper[j, 0] + cutoff_combined
        and box_upper[i, 0] >= box_lower[j, 0]) - cutoff_combined
        and (box_lower[i, 1] <= box_upper[j, 1]+ cutoff_combined
        and box_upper[i, 1] >= box_lower[j, 1]) - cutoff_combined
        and (box_lower[i, 2] <= box_upper[j, 2]+ cutoff_combined
        and box_upper[i, 2] >= box_lower[j, 2]) - cutoff_combined
      ):
        pairs.append((i, j))
  return pairs




def test_sap_broadphase():
  # Create random bounding boxes in min-max format
  ngeom = 10
  # Generate random centers and sizes
  centers = np.random.rand(ngeom, 3) * 3.0
  sizes = np.random.rand(ngeom, 3) * 2.0  # box half-extent up to 1.0 in each direction
  geom_bounding_box_lower = centers - sizes
  geom_bounding_box_upper = centers + sizes

  np_geom_cutoff = np.zeros(ngeom, dtype=np.float32)
  np_collision_group = np.ones(ngeom, dtype=np.int32)

  pairs_np = find_overlapping_pairs_np(geom_bounding_box_lower, geom_bounding_box_upper,
                                        np_geom_cutoff, np_collision_group)
  # print(pairs_np)

  # The number of elements in the lower triangular part of an n x n matrix (excluding the diagonal)
  # is given by n * (n - 1) // 2
  num_lower_tri_elements = ngeom * (ngeom - 1) // 2

  geom_bounding_box_lower_wp = wp.array(geom_bounding_box_lower, dtype=wp.vec3)
  geom_bounding_box_upper_wp = wp.array(geom_bounding_box_upper, dtype=wp.vec3)  
  geom_cutoff = wp.array(np_geom_cutoff)
  collision_group = wp.array(np_collision_group)
  num_candidate_pair = wp.array(
    [
      0,
    ],
    dtype=wp.int32,
  )
  max_candidate_pair = num_lower_tri_elements
  candidate_pair = wp.array(np.zeros((max_candidate_pair, 2), dtype=wp.int32), dtype=wp.vec2i)

  # Temp memory arrays needed for sap_broadphase
  # Factor 2 in some arrays is required for radix sort
  sap_projection_lower_out = wp.array(np.zeros(2*ngeom, dtype=np.float32))
  sap_projection_upper_out = wp.array(np.zeros(ngeom, dtype=np.float32))
  sap_sort_index_out = wp.array(np.zeros(2*ngeom, dtype=np.int32))
  collision_group_tmp = wp.array(np.zeros(2*ngeom, dtype=np.int32))
  index_tracking_tmp = wp.array(np.zeros(ngeom, dtype=np.int32))
  index_tracking_tmp_indexer = wp.array(np.zeros(1, dtype=np.int32))
  sap_range_out = wp.array(np.zeros(ngeom, dtype=np.int32))
  sap_cumulative_sum = wp.array(np.zeros(ngeom, dtype=np.int32))

  sap_broadphase(
    geom_bounding_box_lower_wp,
    geom_bounding_box_upper_wp,
    ngeom,
    geom_cutoff,
    collision_group,
    candidate_pair,
    num_candidate_pair,
    max_candidate_pair,
    sap_projection_lower_out,
    sap_projection_upper_out,
    sap_sort_index_out,
    collision_group_tmp,
    index_tracking_tmp,
    index_tracking_tmp_indexer,
    sap_range_out,
    sap_cumulative_sum,
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
  test_sap_broadphase()
