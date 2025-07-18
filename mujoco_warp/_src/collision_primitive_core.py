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
from .math import normalize_with_norm
from .math import closest_segment_to_segment_points
from .types import MJ_MINVAL
from .math import closest_segment_point

wp.set_module_options({"enable_backward": False})


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


@wp.func
def get_tangent(frame: wp.mat33) -> wp.vec3:
  return wp.vec3(frame[0, 1], frame[1, 1], frame[2, 1])

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
def plane_sphere_core(
  plane: GeomCore,
  sphere: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
) -> int:
  """Calculates one contact between a plane and a sphere."""
  plane_normal = get_plane_normal(plane.rot)
  dist, pos = _plane_sphere(plane_normal, plane.pos, sphere.pos, sphere.size[0])
  contacts[0] = pack_contact_auto_tangent(pos, plane_normal, dist)
  return 1


@wp.func
def _sphere_sphere(
  # In:
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
):
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  return pack_contact_auto_tangent(pos, n, dist)


@wp.func
def _sphere_sphere_ext(
  # In:
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
  mat1: wp.mat33,
  mat2: wp.mat33,
):
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    # Use cross product of z axes like MuJoCo
    axis1 = wp.vec3(mat1[0, 2], mat1[1, 2], mat1[2, 2])
    axis2 = wp.vec3(mat2[0, 2], mat2[1, 2], mat2[2, 2])
    n = wp.cross(axis1, axis2)
    n = wp.normalize(n)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  return pack_contact_auto_tangent(pos, n, dist)


@wp.func
def sphere_sphere_core(
  sphere1: GeomCore,
  sphere2: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
) -> int:
  """Calculates one contact between two spheres."""
  contact = _sphere_sphere(
    sphere1.pos,
    sphere1.size[0],
    sphere2.pos,
    sphere2.size[0],
  )
  contacts[0] = contact
  return 1


@wp.func
def sphere_capsule_core(
  sphere: GeomCore,
  cap: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
) -> int:
  """Calculates one contact between a sphere and a capsule."""
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  length = cap.size[1]
  segment = axis * length

  # Find closest point on capsule centerline to sphere center
  pt = closest_segment_point(cap.pos - segment, cap.pos + segment, sphere.pos)

  # Treat as sphere-sphere collision between sphere and closest point
  contact = _sphere_sphere_ext(
    sphere.pos,
    sphere.size[0],
    pt,
    cap.size[0],
    sphere.rot,
    cap.rot,
  )
  contacts[0] = contact
  return 1


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
  contacts: wp.array(dtype=ContactPoint),
) -> int:
  """Calculates one contact between a plane and an ellipsoid."""
  plane_normal = get_plane_normal(plane.rot)
  sphere_support = -wp.normalize(wp.cw_mul(wp.transpose(ellipsoid.rot) @ plane_normal, ellipsoid.size))
  pos = ellipsoid.pos + ellipsoid.rot @ wp.cw_mul(sphere_support, ellipsoid.size)
  dist = wp.dot(plane_normal, pos - plane.pos)
  contact_pos = pos - plane_normal * dist * 0.5

  contacts[0] = pack_contact_auto_tangent(contact_pos, plane_normal, dist)
  return 1


@wp.func
def capsule_capsule_core(
  cap1: GeomCore,
  cap2: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
) -> int:
  """Calculates one contact between two capsules."""
  axis1 = wp.vec3(cap1.rot[0, 2], cap1.rot[1, 2], cap1.rot[2, 2])
  axis2 = wp.vec3(cap2.rot[0, 2], cap2.rot[1, 2], cap2.rot[2, 2])
  length1 = cap1.size[1]
  length2 = cap2.size[1]
  seg1 = axis1 * length1
  seg2 = axis2 * length2

  pt1, pt2 = closest_segment_to_segment_points(
    cap1.pos - seg1,
    cap1.pos + seg1,
    cap2.pos - seg2,
    cap2.pos + seg2,
  )

  contacts[0] = _sphere_sphere(
    pt1,
    cap1.size[0],
    pt2,
    cap2.size[0],
  )
  return 1


@wp.func
def sphere_cylinder_core(
  sphere: GeomCore,
  cylinder: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
) -> int:
  """Calculates one contact between a sphere and a cylinder."""
  axis = wp.vec3(
    cylinder.rot[0, 2],
    cylinder.rot[1, 2],
    cylinder.rot[2, 2],
  )

  vec = sphere.pos - cylinder.pos
  x = wp.dot(vec, axis)

  a_proj = axis * x
  p_proj = vec - a_proj
  p_proj_sqr = wp.dot(p_proj, p_proj)

  collide_side = wp.abs(x) < cylinder.size[1]
  collide_cap = p_proj_sqr < (cylinder.size[0] * cylinder.size[0])

  if collide_side and collide_cap:
    dist_cap = cylinder.size[1] - wp.abs(x)
    dist_radius = cylinder.size[0] - wp.sqrt(p_proj_sqr)

    if dist_cap < dist_radius:
      collide_side = False
    else:
      collide_cap = False

  # Side collision
  if collide_side:
    pos_target = cylinder.pos + a_proj
    contacts[0] = _sphere_sphere_ext(
      sphere.pos,
      sphere.size[0],
      pos_target,
      cylinder.size[0],
      sphere.rot,
      cylinder.rot,
    )
    return 1

  # Cap collision
  if collide_cap:
    if x > 0.0:
      # top cap
      pos_cap = cylinder.pos + axis * cylinder.size[1]
      plane_normal = axis
    else:
      # bottom cap
      pos_cap = cylinder.pos - axis * cylinder.size[1]
      plane_normal = -axis

    dist, pos_contact = _plane_sphere(plane_normal, pos_cap, sphere.pos, sphere.size[0])
    plane_normal = -plane_normal  # Flip normal after position calculation
    contacts[0] = pack_contact_auto_tangent(pos_contact, plane_normal, dist)
    return 1

  # Corner collision
  inv_len = 1.0 / wp.sqrt(p_proj_sqr)
  p_proj = p_proj * (cylinder.size[0] * inv_len)

  cap_offset = axis * (wp.sign(x) * cylinder.size[1])
  pos_corner = cylinder.pos + cap_offset + p_proj

  contacts[0] = _sphere_sphere_ext(
    sphere.pos,
    sphere.size[0],
    pos_corner,
    0.0,
    sphere.rot,
    cylinder.rot,
  )
  return 1


@wp.func
def _sphere_box(
  # In:
  sphere_pos: wp.vec3,
  sphere_size: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
  margin: float,
):
  center = wp.transpose(box_rot) @ (sphere_pos - box_pos)

  clamped = wp.max(-box_size, wp.min(box_size, center))
  clamped_dir, dist = normalize_with_norm(clamped - center)

  if dist - sphere_size > margin:
    return ContactPoint(), False

  # sphere center inside box
  if dist <= MJ_MINVAL:
    closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])
    k = wp.int32(0)
    for i in range(6):
      face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i // 2] - center[i // 2])
      if closest > face_dist:
        closest = face_dist
        k = i

    nearest = wp.vec3(0.0)
    nearest[k // 2] = wp.where(k % 2, -1.0, 1.0)
    pos = center + nearest * (sphere_size - closest) / 2.0
    contact_normal = box_rot @ nearest
    contact_dist = -closest - sphere_size

  else:
    deepest = center + clamped_dir * sphere_size
    pos = 0.5 * (clamped + deepest)
    contact_normal = box_rot @ clamped_dir
    contact_dist = dist - sphere_size

  contact_pos = box_pos + box_rot @ pos
  contact = pack_contact_auto_tangent(contact_pos, contact_normal, contact_dist)

  return contact, True


@wp.func
def sphere_box_core(
  sphere: GeomCore,
  box: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
  margin: float,
) -> int:
  """Calculates one contact between a sphere and a box."""
  contact, found = _sphere_box(
    sphere.pos,
    sphere.size[0],
    box.pos,
    box.rot,
    box.size,
    margin,
  )

  num_contacts = 0
  if found:
    contacts[0] = contact
    num_contacts = 1

  return num_contacts


@wp.func
def capsule_box_core(
  cap: GeomCore,
  box: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
  margin: float,
) -> int:
  """Calculates contacts between a capsule and a box."""
  # Based on the mjc implementation
  boxmatT = wp.transpose(box.rot)
  pos = boxmatT @ (cap.pos - box.pos)
  axis = boxmatT @ wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  halfaxis = axis * cap.size[1]  # halfaxis is the capsule direction
  axisdir = wp.int32(halfaxis[0] > 0.0) + 2 * wp.int32(halfaxis[1] > 0.0) + 4 * wp.int32(halfaxis[2] > 0.0)

  bestdistmax = margin + 2.0 * (cap.size[0] + cap.size[1] + box.size[0] + box.size[1] + box.size[2])

  # keep track of closest point
  bestdist = wp.float32(bestdistmax)
  bestsegmentpos = wp.float32(-12)

  # cltype: encoded collision configuration
  # cltype / 3 == 0 : lower corner is closest to the capsule
  #            == 2 : upper corner is closest to the capsule
  #            == 1 : middle of the edge is closest to the capsule
  # cltype % 3 == 0 : lower corner is closest to the box
  #            == 2 : upper corner is closest to the box
  #            == 1 : middle of the capsule is closest to the box
  cltype = wp.int32(-4)

  # clface: index of the closest face of the box to the capsule
  # -1: no face is closest (edge or corner is closest)
  # 0, 1, 2: index of the axis perpendicular to the closest face
  clface = wp.int32(-12)

  # first: consider cases where a face of the box is closest
  for i in range(-1, 2, 2):
    axisTip = pos + wp.float32(i) * halfaxis
    boxPoint = wp.vec3(axisTip)

    n_out = wp.int32(0)
    ax_out = wp.int32(-1)

    for j in range(3):
      if boxPoint[j] < -box.size[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = -box.size[j]
      elif boxPoint[j] > box.size[j]:
        n_out += 1
        ax_out = j
        boxPoint[j] = box.size[j]

    if n_out > 1:
      continue

    dist = wp.length_sq(boxPoint - axisTip)

    if dist < bestdist:
      bestdist = dist
      bestsegmentpos = wp.float32(i)
      cltype = -2 + i
      clface = ax_out

  # second: consider cases where an edge of the box is closest
  clcorner = wp.int32(-123)  # which corner is the closest
  cledge = wp.int32(-123)  # which axis
  bestboxpos = wp.float32(0.0)

  for i in range(8):
    for j in range(3):
      if i & (1 << j) != 0:
        continue

      c2 = wp.int32(-123)

      # box_pt is the starting point (corner) on the box
      box_pt = wp.cw_mul(
        wp.vec3(
          wp.where(i & 1, 1.0, -1.0),
          wp.where(i & 2, 1.0, -1.0),
          wp.where(i & 4, 1.0, -1.0),
        ),
        box.size,
      )
      box_pt[j] = 0.0

      # find closest point between capsule and the edge
      dif = box_pt - pos

      u = -box.size[j] * dif[j]
      v = wp.dot(halfaxis, dif)
      ma = box.size[j] * box.size[j]
      mb = -box.size[j] * halfaxis[j]
      mc = cap.size[1] * cap.size[1]
      det = ma * mc - mb * mb
      if wp.abs(det) < MJ_MINVAL:
        continue

      idet = 1.0 / det
      # sX : X=1 means middle of segment. X=0 or 2 one or the other end

      x1 = wp.float32((mc * u - mb * v) * idet)
      x2 = wp.float32((ma * v - mb * u) * idet)

      s1 = wp.int32(1)
      s2 = wp.int32(1)

      if x1 > 1:
        x1 = 1.0
        s1 = 2
        x2 = (v - mb) / mc
      elif x1 < -1:
        x1 = -1.0
        s1 = 0
        x2 = (v + mb) / mc

      x2_over = x2 > 1.0
      if x2_over or x2 < -1.0:
        if x2_over:
          x2 = 1.0
          s2 = 2
          x1 = (u - mb) / ma
        else:
          x2 = -1.0
          s2 = 0
          x1 = (u + mb) / ma

        if x1 > 1:
          x1 = 1.0
          s1 = 2
        elif x1 < -1:
          x1 = -1.0
          s1 = 0

      dif -= halfaxis * x2
      dif[j] += box.size[j] * x1

      # encode relative positions of the closest points
      ct = s1 * 3 + s2

      dif_sq = wp.length_sq(dif)
      if dif_sq < bestdist - MJ_MINVAL:
        bestdist = dif_sq
        bestsegmentpos = x2
        bestboxpos = x1
        # ct<6 means closest point on box is at lower end or middle of edge
        c2 = ct / 6

        clcorner = i + (1 << j) * c2  # index of closest box corner
        cledge = j  # axis index of closest box edge
        cltype = ct  # encoded collision configuration

  best = wp.float32(0.0)

  p = wp.vec2(pos.x, pos.y)
  dd = wp.vec2(halfaxis.x, halfaxis.y)
  s = wp.vec2(box.size.x, box.size.y)
  secondpos = wp.float32(-4.0)

  uu = dd.x * s.y
  vv = dd.y * s.x
  w_neg = dd.x * p.y - dd.y * p.x < 0

  best = wp.float32(-1.0)

  ee1 = uu - vv
  ee2 = uu + vv

  if wp.abs(ee1) > best:
    best = wp.abs(ee1)
    c1 = wp.where((ee1 < 0) == w_neg, 0, 3)

  if wp.abs(ee2) > best:
    best = wp.abs(ee2)
    c1 = wp.where((ee2 > 0) == w_neg, 1, 2)

  if cltype == -4:  # invalid type
    return 0

  if cltype >= 0 and cltype / 3 != 1:  # closest to a corner of the box
    c1 = axisdir ^ clcorner
    # Calculate relative orientation between capsule and corner
    # There are two possible configurations:
    # 1. Capsule axis points toward/away from corner
    # 2. Capsule axis aligns with a face or edge
    if c1 != 0 and c1 != 7:  # create second contact point
      if c1 == 1 or c1 == 2 or c1 == 4:
        mul = 1
      else:
        mul = -1
        c1 = 7 - c1

      # "de" and "dp" distance from first closest point on the capsule to both ends of it
      # mul is a direction along the capsule's axis

      if c1 == 1:
        ax = 0
        ax1 = 1
        ax2 = 2
      elif c1 == 2:
        ax = 1
        ax1 = 2
        ax2 = 0
      elif c1 == 4:
        ax = 2
        ax1 = 0
        ax2 = 1

      if axis[ax] * axis[ax] > 0.5:  # second point along the edge of the box
        m = 2.0 * box.size[ax] / wp.abs(halfaxis[ax])
        secondpos = min(1.0 - wp.float32(mul) * bestsegmentpos, m)
      else:  # second point along a face of the box
        # check for overshoot again
        m = 2.0 * min(
          box.size[ax1] / wp.abs(halfaxis[ax1]),
          box.size[ax2] / wp.abs(halfaxis[ax2]),
        )
        secondpos = -min(1.0 + wp.float32(mul) * bestsegmentpos, m)
      secondpos *= wp.float32(mul)

  elif cltype >= 0 and cltype / 3 == 1:  # we are on box's edge
    # Calculate relative orientation between capsule and edge
    # Two possible configurations:
    # - T configuration: c1 = 2^n (no additional contacts)
    # - X configuration: c1 != 2^n (potential additional contacts)
    c1 = axisdir ^ clcorner
    c1 &= 7 - (1 << cledge)  # mask out edge axis to determine configuration

    if c1 == 1 or c1 == 2 or c1 == 4:  # create second contact point
      if cledge == 0:
        ax1 = 1
        ax2 = 2
      if cledge == 1:
        ax1 = 2
        ax2 = 0
      if cledge == 2:
        ax1 = 0
        ax2 = 1
      ax = cledge

      # find which face the capsule has a lower angle, and switch the axis
      if wp.abs(axis[ax1]) > wp.abs(axis[ax2]):
        ax1 = ax2
      ax2 = 3 - ax - ax1

      # mul determines direction along capsule axis for second contact point
      if c1 & (1 << ax2):
        mul = 1
        secondpos = 1.0 - bestsegmentpos
      else:
        mul = -1
        secondpos = 1.0 + bestsegmentpos

      # now find out whether we point towards the opposite side or towards one of the sides
      # and also find the farthest point along the capsule that is above the box

      e1 = 2.0 * box.size[ax2] / wp.abs(halfaxis[ax2])
      secondpos = min(e1, secondpos)

      if ((axisdir & (1 << ax)) != 0) == ((c1 & (1 << ax2)) != 0):
        e2 = 1.0 - bestboxpos
      else:
        e2 = 1.0 + bestboxpos

      e1 = box.size[ax] * e2 / wp.abs(halfaxis[ax])

      secondpos = min(e1, secondpos)
      secondpos *= wp.float32(mul)

  elif cltype < 0:
    # similarly we handle the case when one capsule's end is closest to a face of the box
    # and find where is the other end pointing to and clamping to the farthest point
    # of the capsule that's above the box
    # if the closest point is inside the box there's no need for a second point

    if clface != -1:  # create second contact point
      mul = wp.where(cltype == -3, 1, -1)
      secondpos = 2.0

      tmp1 = pos - halfaxis * wp.float32(mul)

      for i in range(3):
        if i != clface:
          ha_r = wp.float32(mul) / halfaxis[i]
          e1 = (box.size[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1

          e1 = (-box.size[i] - tmp1[i]) * ha_r
          if 0 < e1 and e1 < secondpos:
            secondpos = e1

      secondpos *= wp.float32(mul)

  num_contacts = 0
  # create sphere in original orientation at first contact point
  s1_pos_l = pos + halfaxis * bestsegmentpos
  s1_pos_g = box.rot @ s1_pos_l + box.pos

  # collide with sphere
  contact, found = _sphere_box(
    s1_pos_g,
    cap.size[0],
    box.pos,
    box.rot,
    box.size,
    margin,
  )
  if found:
    contacts[num_contacts] = contact
    num_contacts += 1

  if secondpos > -3:  # secondpos was modified
    s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)
    s2_pos_g = box.rot @ s2_pos_l + box.pos
    contact, found = _sphere_box(
      s2_pos_g,
      cap.size[0],
      box.pos,
      box.rot,
      box.size,
      margin,
    )
    if found:
      contacts[num_contacts] = contact
      num_contacts += 1

  return num_contacts


@wp.func
def plane_box_core(
  plane: GeomCore,
  box: GeomCore,
  contacts: wp.array(dtype=ContactPoint),
  margin: float,
) -> int:
  """Calculates contacts between a plane and a box.

  Can generate up to 4 contact points for the penetrating corners.

  Returns:
      int: Number of contacts generated (0-4)
  """
  num_contacts = int(0)
  plane_normal = get_plane_normal(plane.rot)
  corner = wp.vec3()
  dist = wp.dot(box.pos - plane.pos, plane_normal)

  # test all corners, pick bottom 4
  for i in range(8):
    # get corner in local coordinates
    corner.x = wp.where(i & 1, box.size.x, -box.size.x)
    corner.y = wp.where(i & 2, box.size.y, -box.size.y)
    corner.z = wp.where(i & 4, box.size.z, -box.size.z)

    # get corner in global coordinates relative to box center
    corner = box.rot @ corner

    # compute distance to plane, skip if too far or pointing up
    ldist = wp.dot(plane_normal, corner)
    if dist + ldist > margin or ldist > 0:
      continue

    cdist = dist + ldist
    contact_pos = corner + box.pos - plane_normal * (cdist * 0.5)

    contacts[num_contacts] = pack_contact_auto_tangent(
      contact_pos, plane_normal, cdist
    )
    num_contacts += 1

    if num_contacts >= 4:
      break

  return num_contacts



@wp.kernel
def dummy_test(a: GeomCore, b: GeomCore):
  contacts = wp.zeros(shape=(8,), dtype=ContactPoint)

  num_ellipsoid_contacts = plane_ellipsoid_core(a, b, contacts)
  num_capsule_contacts = plane_capsule_core(a, b, contacts)
  num_cylinder_contacts = plane_cylinder_core(a, b, contacts, 1.0)


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
