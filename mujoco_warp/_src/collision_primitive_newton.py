import warp as wp


from .math import closest_segment_point
from .math import closest_segment_to_segment_points
from .math import make_frame
from .math import normalize_with_norm

wp.config.enable_backward = False

wp.clear_kernel_cache()


@wp.struct
class Geom:
  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  vertadr: int
  vertnum: int
  vert: wp.array(dtype=wp.vec3)


@wp.struct
class ContactFrame:
  pos: wp.vec3
  frame: wp.mat33  # The first row of the frame is the normal
  dist: float


@wp.func
def _plane_sphere(plane_normal: wp.vec3, plane_pos: wp.vec3, sphere_pos: wp.vec3, sphere_radius: float):
  dist = wp.dot(sphere_pos - plane_pos, plane_normal) - sphere_radius
  pos = sphere_pos - plane_normal * (sphere_radius + 0.5 * dist)
  return dist, pos


@wp.func
def plane_sphere(
  # In:
  plane: Geom,
  sphere: Geom,
) -> ContactFrame:
  dist, pos = _plane_sphere(plane.normal, plane.pos, sphere.pos, sphere.size[0])

  # Return contact frame using make_frame helper
  return ContactFrame(pos=pos, frame=make_frame(plane.normal), dist=dist)


@wp.func
def _sphere_sphere(pos1: wp.vec3, radius1: float, pos2: wp.vec3, radius2: float) -> ContactFrame:
  dir = pos2 - pos1
  dist = wp.length(dir)
  if dist == 0.0:
    n = wp.vec3(1.0, 0.0, 0.0)
  else:
    n = dir / dist
  dist = dist - (radius1 + radius2)
  pos = pos1 + n * (radius1 + 0.5 * dist)

  return ContactFrame(pos=pos, frame=make_frame(n), dist=dist)


@wp.func
def sphere_sphere(
  # In:
  sphere1: Geom,
  sphere2: Geom,
) -> ContactFrame:
  return _sphere_sphere(
    sphere1.pos,
    sphere1.size[0],
    sphere2.pos,
    sphere2.size[0],
  )


@wp.func
def sphere_capsule(
  # In:
  sphere: Geom,
  cap: Geom,
) -> ContactFrame:
  """Calculates one contact between a sphere and a capsule."""
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  length = cap.size[1]
  segment = axis * length

  # Find closest point on capsule centerline to sphere center
  pt = closest_segment_point(cap.pos - segment, cap.pos + segment, sphere.pos)

  # Treat as sphere-sphere collision between sphere and closest point
  return _sphere_sphere(
    sphere.pos,
    sphere.size[0],
    pt,
    cap.size[0],
  )


@wp.func
def capsule_capsule(
  # In:
  cap1: Geom,
  cap2: Geom,
) -> ContactFrame:
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

  return _sphere_sphere(
    pt1,
    cap1.size[0],
    pt2,
    cap2.size[0],
  )


@wp.func
def plane_capsule(
  # In:
  plane: Geom,
  cap: Geom,
):
  """Calculates two contacts between a capsule and a plane."""
  n = plane.normal
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  # align contact frames with capsule axis
  b, b_norm = normalize_with_norm(axis - n * wp.dot(n, axis))

  if b_norm < 0.5:
    if -0.5 < n[1] and n[1] < 0.5:
      b = wp.vec3(0.0, 1.0, 0.0)
    else:
      b = wp.vec3(0.0, 0.0, 1.0)

  c = wp.cross(n, b)
  frame = wp.mat33(n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2])
  segment = axis * cap.size[1]

  dist1, pos1 = _plane_sphere(n, plane.pos, cap.pos + segment, cap.size[0])
  dist2, pos2 = _plane_sphere(n, plane.pos, cap.pos - segment, cap.size[0])

  return ContactFrame(pos=pos1, frame=frame, dist=dist1), ContactFrame(pos=pos2, frame=frame, dist=dist2)


@wp.func
def _sphere_sphere_ext(
  # In:
  pos1: wp.vec3,
  radius1: float,
  pos2: wp.vec3,
  radius2: float,
  mat1: wp.mat33,
  mat2: wp.mat33,
) -> ContactFrame:
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

  return ContactFrame(pos=pos, frame=make_frame(n), dist=dist)


@wp.func
def sphere_cylinder(
  # In:
  sphere: Geom,
  cylinder: Geom,
) -> ContactFrame:
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

    return _sphere_sphere_ext(
      sphere.pos,
      sphere.size[0],
      pos_target,
      cylinder.size[0],
      sphere.rot,
      cylinder.rot,
    )

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

    return ContactFrame(pos=pos_contact, frame=make_frame(plane_normal), dist=dist)

  # Corner collision
  inv_len = 1.0 / wp.sqrt(p_proj_sqr)
  p_proj = p_proj * (cylinder.size[0] * inv_len)

  cap_offset = axis * (wp.sign(x) * cylinder.size[1])
  pos_corner = cylinder.pos + cap_offset + p_proj

  return _sphere_sphere_ext(
    sphere.pos,
    sphere.size[0],
    pos_corner,
    0.0,
    sphere.rot,
    cylinder.rot,
  )


@wp.func
def _sphere_box(
  sphere_pos: wp.vec3, sphere_size: float, box_pos: wp.vec3, box_rot: wp.mat33, box_size: wp.vec3, margin: float
) -> ContactFrame:
  center = wp.transpose(box_rot) @ (sphere_pos - box_pos)

  clamped = wp.max(-box_size, wp.min(box_size, center))
  clamped_dir, dist = normalize_with_norm(clamped - center)

  if dist - sphere_size > margin:
    return ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=100000.0)

  # sphere center inside box
  if dist <= float(1e-8):
    closest = 2.0 * (box_size[0] + box_size[1] + box_size[2])
    k = wp.int32(0)
    for i in range(6):
      face_dist = wp.abs(wp.where(i % 2, 1.0, -1.0) * box_size[i / 2] - center[i / 2])
      if closest > face_dist:
        closest = face_dist
        k = i

    nearest = wp.vec3(0.0)
    nearest[k / 2] = wp.where(k % 2, -1.0, 1.0)
    pos = center + nearest * (sphere_size - closest) / 2.0
    contact_normal = box_rot @ nearest
    contact_dist = -closest - sphere_size

  else:
    deepest = center + clamped_dir * sphere_size
    pos = 0.5 * (clamped + deepest)
    contact_normal = box_rot @ clamped_dir
    contact_dist = dist - sphere_size

  contact_pos = box_pos + box_rot @ pos
  return ContactFrame(pos=contact_pos, frame=make_frame(contact_normal), dist=contact_dist)


@wp.func
def sphere_box(
  # In:
  sphere: Geom,
  box: Geom,
  margin: float,
) -> ContactFrame:
  return _sphere_box(sphere.pos, sphere.size[0], box.pos, box.rot, box.size, margin)


@wp.func
def capsule_box(
  # In:
  cap: Geom,
  box: Geom,
  margin: float,
):
  """Calculates contacts between a capsule and a box."""
  # Based on the mjc implementation
  pos = wp.transpose(box.rot) @ (cap.pos - box.pos)
  axis = wp.vec3(cap.rot[0, 2], cap.rot[1, 2], cap.rot[2, 2])
  halfaxis = axis * cap.size[1]  # halfaxis is the capsule direction
  axisdir = wp.int32(axis[0] > 0.0) + 2 * wp.int32(axis[1] > 0.0) + 4 * wp.int32(axis[2] > 0.0)

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
      if wp.abs(det) < float(1e-8):
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
      if dif_sq < bestdist - float(1e-8):
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
    return ContactFrame(), ContactFrame(), 0

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

      # Then it finds with which face the capsule has a lower angle and switches the axis names
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

      # now we have to find out whether we point towards the opposite side or towards one of the
      # sides and also find the farthest point along the capsule that is above the box

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

  # create sphere in original orientation at first contact point
  s1_pos_l = pos + halfaxis * bestsegmentpos
  s1_pos_g = box.rot @ s1_pos_l + box.pos

  # collide with sphere
  contact = _sphere_box(s1_pos_g, cap.size[0], box.pos, box.rot, box.size, margin)

  if secondpos > -3:  # secondpos was modified
    s2_pos_l = pos + halfaxis * (secondpos + bestsegmentpos)
    s2_pos_g = box.rot @ s2_pos_l + box.pos
    contact2 = _sphere_box(s2_pos_g, cap.size[0], box.pos, box.rot, box.size, margin)

    return contact, contact2, 2

  return contact, ContactFrame(), 1


@wp.func
def plane_box(
  # In:
  plane: Geom,
  box: Geom,
  margin: float,
):
  count = int(0)
  corner = wp.vec3()
  dist = wp.dot(box.pos - plane.pos, plane.normal)

  # Initialize contact frames
  contact1 = ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=0.0)
  contact2 = ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=0.0)
  contact3 = ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=0.0)
  contact4 = ContactFrame(pos=wp.vec3(0.0), frame=wp.mat33(1.0), dist=0.0)
  count = int(0)

  # test all corners, pick bottom 4
  for i in range(8):
    # get corner in local coordinates
    corner.x = wp.where(i & 1, box.size.x, -box.size.x)
    corner.y = wp.where(i & 2, box.size.y, -box.size.y)
    corner.z = wp.where(i & 4, box.size.z, -box.size.z)

    # get corner in global coordinates relative to box center
    corner = box.rot * corner

    # compute distance to plane, skip if too far or pointing up
    ldist = wp.dot(plane.normal, corner)
    if dist + ldist > margin or ldist > 0:
      continue

    cdist = dist + ldist
    frame = make_frame(plane.normal)
    pos = corner + box.pos + (plane.normal * cdist / -2.0)

    if count == 0:
      contact1 = ContactFrame(pos=pos, frame=frame, dist=cdist)
    elif count == 1:
      contact2 = ContactFrame(pos=pos, frame=frame, dist=cdist)
    elif count == 2:
      contact3 = ContactFrame(pos=pos, frame=frame, dist=cdist)
    elif count == 3:
      contact4 = ContactFrame(pos=pos, frame=frame, dist=cdist)

    count += 1
    if count >= 4:
      break

  return contact1, contact2, contact3, contact4, count


_HUGE_VAL = 1e6


@wp.func
def plane_convex(
  # In:
  plane: Geom,
  convex: Geom,
):
  """Calculates contacts between a plane and a convex object."""

  # get points in the convex frame
  plane_pos = wp.transpose(convex.rot) @ (plane.pos - convex.pos)
  n = wp.transpose(convex.rot) @ plane.normal

  # Find support points
  max_support = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)

    max_support = wp.max(support, max_support)

  threshold = wp.max(0.0, max_support - 1e-3)

  # Store indices in vec4
  indices = wp.vec4i(-1, -1, -1, -1)

  # TODO(team): Explore faster methods like tile_min or even fast pass kernels if the upper bound of vertices in all convexes is small enough such that all vertices fit into shared memory
  # Find point a (first support point)
  a_dist = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)
    dist = wp.where(support > threshold, 0.0, -_HUGE_VAL)
    if dist > a_dist:
      indices[0] = i
      a_dist = dist
  a = convex.vert[convex.vertadr + indices[0]]

  # Find point b (furthest from a)
  b_dist = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)
    dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
    dist = wp.length_sq(a - convex.vert[convex.vertadr + i]) + dist_mask
    if dist > b_dist:
      indices[1] = i
      b_dist = dist
  b = convex.vert[convex.vertadr + indices[1]]

  # Find point c (furthest along axis orthogonal to a-b)
  ab = wp.cross(n, a - b)
  c_dist = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)
    dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
    ap = a - convex.vert[convex.vertadr + i]
    dist = wp.abs(wp.dot(ap, ab)) + dist_mask
    if dist > c_dist:
      indices[2] = i
      c_dist = dist
  c = convex.vert[convex.vertadr + indices[2]]

  # Find point d (furthest from other triangle edges)
  ac = wp.cross(n, a - c)
  bc = wp.cross(n, b - c)
  d_dist = wp.float32(-_HUGE_VAL)
  for i in range(convex.vertnum):
    support = wp.dot(plane_pos - convex.vert[convex.vertadr + i], n)
    dist_mask = wp.where(support > threshold, 0.0, -_HUGE_VAL)
    ap = a - convex.vert[convex.vertadr + i]
    bp = b - convex.vert[convex.vertadr + i]
    dist_ap = wp.abs(wp.dot(ap, ac)) + dist_mask
    dist_bp = wp.abs(wp.dot(bp, bc)) + dist_mask
    if dist_ap + dist_bp > d_dist:
      indices[3] = i
      d_dist = dist_ap + dist_bp

  # Prepare contacts
  frame = make_frame(plane.normal)
  contact1 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact2 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact3 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  contact4 = ContactFrame(pos=wp.vec3(0.0), frame=frame, dist=0.0)
  count = int(0)

  for i in range(3, -1, -1):
    idx = indices[i]
    unique_count = int(0)
    for j in range(i + 1):
      if indices[j] == idx:
        unique_count = unique_count + 1

    # Check if the index is unique (appears exactly once)
    if unique_count == 1:
      pos = convex.vert[convex.vertadr + idx]
      pos = convex.pos + convex.rot @ pos
      support = wp.dot(plane_pos - convex.vert[convex.vertadr + idx], n)
      dist = -support
      pos = pos - 0.5 * dist * plane.normal

      if count == 0:
        contact1 = ContactFrame(pos=pos, frame=frame, dist=dist)
      elif count == 1:
        contact2 = ContactFrame(pos=pos, frame=frame, dist=dist)
      elif count == 2:
        contact3 = ContactFrame(pos=pos, frame=frame, dist=dist)
      elif count == 3:
        contact4 = ContactFrame(pos=pos, frame=frame, dist=dist)
      count += 1

  return contact1, contact2, contact3, contact4, count
