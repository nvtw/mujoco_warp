import warp as wp



wp.config.enable_backward = False





from .collision_primitive import Geom
from .types import MJ_MINVAL
from .types import GeomType





# newton/geometry/types.py (or within geometry/__init__.py)

# Shape geometry types
GEO_SPHERE = wp.constant(0)
GEO_BOX = wp.constant(1)
GEO_CAPSULE = wp.constant(2)
GEO_CYLINDER = wp.constant(3)
GEO_CONE = wp.constant(4)
GEO_MESH = wp.constant(5)
GEO_SDF = wp.constant(6)
GEO_PLANE = wp.constant(7)
GEO_NONE = wp.constant(8)
GEO_CONVEX = wp.constant(9)




FLOAT_MIN = -1e30
FLOAT_MAX = 1e30
EPS_BEST_COUNT = 12
MULTI_CONTACT_COUNT = 4
MULTI_POLYGON_COUNT = 8
MULTI_TILT_ANGLE = 1.0

matc3 = wp.types.matrix(shape=(EPS_BEST_COUNT, 3), dtype=float)
vecc3 = wp.types.vector(EPS_BEST_COUNT * 3, dtype=float)

# Matrix definition for the `tris` scratch space which is used to store the
# triangles of the polytope. Note that the first dimension is 2, as we need
# to store the previous and current polytope. But since Warp doesn't support
# 3D matrices yet, we use 2 * 3 * EPS_BEST_COUNT as the first dimension.
TRIS_DIM = 3 * EPS_BEST_COUNT
mat2c3 = wp.types.matrix(shape=(2 * TRIS_DIM, 3), dtype=float)
mat3p = wp.types.matrix(shape=(MULTI_POLYGON_COUNT, 3), dtype=float)
mat3c = wp.types.matrix(shape=(MULTI_CONTACT_COUNT, 3), dtype=float)
mat43 = wp.types.matrix(shape=(4, 3), dtype=float)

vec6 = wp.types.vector(6, dtype=int)
VECI1 = vec6(0, 0, 0, 1, 1, 2)
VECI2 = vec6(1, 2, 3, 2, 3, 3)




@wp.func
def gjk_support_geom(geom: Geom, geomtype: int, dir: wp.vec3, verts: wp.array(dtype=wp.vec3)):
  local_dir = wp.transpose(geom.rot) @ dir
  if geomtype == int(GeomType.SPHERE.value):
    support_pt = geom.pos + geom.size[0] * dir
  elif geomtype == int(GeomType.BOX.value):
    res = wp.cw_mul(wp.sign(local_dir), geom.size)
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.CAPSULE.value):
    res = local_dir * geom.size[0]
    # add cylinder contribution
    res[2] += wp.sign(local_dir[2]) * geom.size[1]
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.ELLIPSOID.value):
    res = wp.cw_mul(local_dir, geom.size)
    res = wp.normalize(res)
    # transform to ellipsoid
    res = wp.cw_mul(res, geom.size)
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.CYLINDER.value):
    res = wp.vec3(0.0, 0.0, 0.0)
    # set result in XY plane: support on circle
    d = wp.sqrt(wp.dot(local_dir, local_dir))
    if d > MJ_MINVAL:
      scl = geom.size[0] / d
      res[0] = local_dir[0] * scl
      res[1] = local_dir[1] * scl
    # set result in Z direction
    res[2] = wp.sign(local_dir[2]) * geom.size[1]
    support_pt = geom.rot @ res + geom.pos
  elif geomtype == int(GeomType.MESH.value):
    max_dist = float(FLOAT_MIN)
    # exhaustive search over all vertices
    # TODO(team): consider hill-climb over graph data
    for i in range(geom.vertnum):
      vert = verts[geom.vertadr + i]
      dist = wp.dot(vert, local_dir)
      if dist > max_dist:
        max_dist = dist
        support_pt = vert
    support_pt = geom.rot @ support_pt + geom.pos

  return wp.dot(support_pt, dir), support_pt



@wp.func
def gjk_support(
  # In:
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
  dir: wp.vec3,
  verts: wp.array(dtype=wp.vec3),
):
  # Returns the distance between support points on two geoms, and the support point.
  # Negative distance means objects are not intersecting along direction `dir`.
  # Positive distance means objects are intersecting along the given direction `dir`.

  dist1, s1 = gjk_support_geom(geom1, geomtype1, dir, verts)
  dist2, s2 = gjk_support_geom(geom2, geomtype2, -dir, verts)

  support_pt = s1 - s2
  return dist1 + dist2, support_pt