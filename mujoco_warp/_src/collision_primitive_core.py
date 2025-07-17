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

import warp as wp
from typing import Any, Tuple

wp.set_module_options({"enable_backward": False})


@wp.func
def normalize_with_norm(x: Any):
  norm = wp.length(x)
  if norm == 0.0:
    return x, 0.0
  return x / norm, norm


@wp.func
def orthogonals(a: wp.vec3):
  y = wp.vec3(0.0, 1.0, 0.0)
  z = wp.vec3(0.0, 0.0, 1.0)
  b = wp.where((-0.5 < a[1]) and (a[1] < 0.5), y, z)
  b = b - a * wp.dot(a, b)
  b = wp.normalize(b)
  if wp.length(a) == 0.0:
    b = wp.vec3(0.0, 0.0, 0.0)
  c = wp.cross(a, b)

  return b, c


@wp.func
def make_tangent(a: wp.vec3):
  a = wp.normalize(a)
  b, c = orthogonals(a)
  return b


@wp.struct
class ContactPoint:
  pos: wp.vec3
  normal: wp.vec3
  tangent: wp.vec3
  dist: float


@wp.func
def pack_contact(pos: wp.vec3, normal: wp.vec3, tangent: wp.vec3, dist: float) -> ContactPoint:
  return ContactPoint(pos=pos, normal=normal, tangent=tangent, dist=dist)


@wp.func
def pack_contact_auto_tangent(pos: wp.vec3, normal: wp.vec3, dist: float) -> ContactPoint:
  tangent = make_tangent(normal)
  return ContactPoint(pos=pos, normal=normal, tangent=tangent, dist=dist)


@wp.func
def extract_frame(c: ContactPoint) -> wp.mat33:
  normal = c.normal
  tangent = c.tangent
  tangent2 = wp.cross(normal, tangent)
  return wp.mat33(normal[0], normal[1], normal[2], tangent[0], tangent[1], tangent[2], tangent2[0], tangent2[1], tangent2[2])


@wp.struct
class GeomCore:
  pos: wp.vec3
  rot: wp.mat33
  size: wp.vec3


@wp.func
def get_plane_normal(rot: wp.mat33) -> wp.vec3:
  return wp.vec3(rot[0, 2], rot[1, 2], rot[2, 2])


@wp.func
def geom_core(
  # Data in:
  geom_xpos_in: wp.array2d(dtype=wp.vec3),
  geom_xmat_in: wp.array2d(dtype=wp.mat33),
  geom_size: wp.array2d(dtype=wp.vec3),
  # In:
  worldid: int,
  gid: int,
) -> GeomCore:
  geom = GeomCore()
  geom.pos = geom_xpos_in[worldid, gid]
  rot = geom_xmat_in[worldid, gid]
  geom.rot = rot
  geom.size = geom_size[worldid, gid]
  return geom


@wp.func
def _plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float):
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def plane_capsule_core(
  plane: GeomCore,
  cap: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
) -> int:
  """Calculates two contacts between a capsule and a plane.

  Finds contact points at both ends of the capsule where they intersect with the plane.
  The contact normal is aligned with the plane normal.

  Returns:
      int: Always returns 2 since there are two contact points (one at each end)
  """
  n = get_plane_normal(plane.rot)
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  # align contact frames with capsule axis
  b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))

  if b_norm < 0.5:
    if -0.5 < n[1] and n[1] < 0.5:
      b = wp.vec3(0.0, 1.0, 0.0)
    else:
      b = wp.vec3(0.0, 0.0, 1.0)

  # c = wp.cross(n, b)
  # frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])
  segment = axis * cap.size[1]

  dist1, pos1 = _plane_sphere(n, plane.pos, cap.pos + segment, cap.size[0])
  contacts[0] = pack_contact(pos1, n, b, dist1)

  dist2, pos2 = _plane_sphere(n, plane.pos, cap.pos - segment, cap.size[0])
  contacts[1] = pack_contact(pos2, n, b, dist2)

  return 2


@wp.func
def plane_ellipsoid_core(
  plane: GeomCore,
  ellipsoid: GeomCore,
) -> ContactPoint:
  """Calculates one contact between a plane and an ellipsoid."""
  plane_normal = get_plane_normal(plane.rot)
  sphere_support = -wp.normalize(wp.cw_mul(wp.transpose(ellipsoid.rot) @ plane_normal, ellipsoid.size))
  pos = ellipsoid.pos + ellipsoid.rot @ wp.cw_mul(sphere_support, ellipsoid.size)
  dist = wp.dot(plane_normal, pos - plane.pos)
  contact_pos = pos - plane_normal * dist * 0.5

  return pack_contact_auto_tangent(contact_pos, plane_normal, dist)


@wp.kernel
def dummy_test(a: GeomCore, b: GeomCore):
  contacts = wp.zeros(shape=(8,), dtype=ContactPoint)

  c = plane_ellipsoid_core(a, b)
  num_contacts = plane_capsule_core(a, b, contacts)


def test():
  # Create dummy geoms
  plane = GeomCore()
  plane.pos = wp.vec3(0.0, 0.0, 0.0)
  plane.rot = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  plane.size = wp.vec3(1.0, 1.0, 1.0)

  ellipsoid = GeomCore()
  ellipsoid.pos = wp.vec3(0.0, 0.0, 1.0)
  ellipsoid.rot = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
  ellipsoid.size = wp.vec3(0.5, 0.5, 0.5)

  # Launch kernel
  wp.launch(kernel=dummy_test, dim=1, inputs=[plane, ellipsoid])


if __name__ == "__main__":
  test()
