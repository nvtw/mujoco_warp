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
  if dist <= MJ_MINVAL:
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
