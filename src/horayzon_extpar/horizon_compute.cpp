#include "horizon_compute.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <embree4/rtcore.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

using namespace std;
using namespace shapes;

// Namespace
#if defined(RTC_NAMESPACE_USE)
	RTC_NAMESPACE_USE
#endif

//-----------------------------------------------------------------------------
// Structures
//-----------------------------------------------------------------------------

/**
 * @brief Indices, unit normals, and unit edge directions of triangles adjacent
 *        to a vertex.
 */
struct AdjacentTriangles {
    unsigned int idx_vert;
    const unsigned int* tri_face;
    std::vector<unsigned int> idx_tri;
    std::vector<geom_vector> tri_normal;  // unit normals of adjacent triangles
    std::vector<geom_vector> tri_edge_0;  // unit directions of the two edges
    std::vector<geom_vector> tri_edge_1;  // of adjacent triangles
};

//-----------------------------------------------------------------------------
// Geometric functions not dependent on Embree
//-----------------------------------------------------------------------------

/**
 * @brief Convert degree to radian.
 * @param angle Input angle [deg].
 * @return Output angle [rad].
 */
inline double deg2rad(double angle) {
	return ((angle / 180.0) * M_PI);
}

/**
 * @brief Convert radian to degree.
 * @param angle Input angle [rad].
 * @return Output angle [deg].
 */
inline double rad2deg(double angle) {
	return ((angle / M_PI) * 180.0);
}

/**
 * @brief Compute the dot product between two vectors.
 * @param a Vector a.
 * @param b Vector b.
 * @return Resulting dot product.
 */
inline double dot_product(geom_vector a, geom_vector b) {
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

/**
 * @brief Compute the cross product between two vectors.
 * @param a Vector a.
 * @param b Vector b.
 * @return Resulting cross product.
 */
inline geom_vector cross_product(geom_vector a, geom_vector b) {
    geom_vector c = {a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x};
    return c;
}

/**
 * @brief Compute the unit vector (normalised vector) of a vector in-place.
 * @param a Vector a.
 */
void unit_vector(geom_vector& a) {
    double vector_mag = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    a.x /= vector_mag;
    a.y /= vector_mag;
    a.z /= vector_mag;
}

/**
 * @brief Rotate vector v around unit vector k with a given angle.
 *
 * This function rotates vector v around a unit vector k with a given angle
 * according to the Rodrigues' rotation formula. For performance reasons,
 * trigonometric function have to be pre-computed.
 *
 * @param v Vector that should be rotated.
 * @param k Unit vector specifying the rotation axis [-].
 * @param angle_rot_sin Sine of the rotation angle.
 * @param angle_rot_cos Cosine of the rotation angle.
 * @return Rotated vector.
 */
inline geom_vector vector_rotation(
    geom_vector v,
    geom_vector k,
    double angle_rot_sin,
    double angle_rot_cos
) {
    geom_vector v_rot;
    double term = dot_product(k, v) * (1.0 - angle_rot_cos);
    v_rot.x = v.x * angle_rot_cos + (k.y * v.z - k.z * v.y) * angle_rot_sin
        + k.x * term;
    v_rot.y = v.y * angle_rot_cos + (k.z * v.x - k.x * v.z) * angle_rot_sin
        + k.y * term;
    v_rot.z = v.z * angle_rot_cos + (k.x * v.y - k.y * v.x) * angle_rot_sin
        + k.z * term;
    return v_rot;
}

/**
 * @brief Compute centroid coordinates of a triangle.
 * @param vert_0 Vertex 0.
 * @param vert_1 Vertex 1.
 * @param vert_2 Vertex 2.
 * @return Centroid.
 */
inline geom_point compute_centroid(
    geom_point vert_0,
    geom_point vert_1,
    geom_point vert_2
) {
    geom_point centroid = {
        (vert_0.x + vert_1.x + vert_2.x) / 3.0,
        (vert_0.y + vert_1.y + vert_2.y) / 3.0,
        (vert_0.z + vert_1.z + vert_2.z) / 3.0
    };
    return centroid;
}

/**
 * @brief Compute triangle normal (unit vector)
 *
 * Compute the normal (unit vector) of a triangle. The direction of the normal
 * vector depends on the winding order of the vertices.
 *
 * @param vert_0 Vertex 0.
 * @param vert_1 Vertex 1.
 * @param vert_2 Vertex 2.
 * @return Triangle normal (unit vector) [-].
 */
inline geom_vector get_triangle_normal(
    geom_point vert_0,
    geom_point vert_1,
    geom_point vert_2
) {
    geom_vector tri_normal = cross_product(vert_2 - vert_1, vert_0 - vert_1);
    unit_vector(tri_normal);
    return tri_normal;
}

/**
 * @brief Compute the minimal dot product between a vector and a set of
 *        vectors.
 * @param d Vector.
 * @param vector_set Set of vectors.
 * @return Minimal dot product.
 */
inline double min_dot_product(
    geom_vector d,
    const std::vector<geom_vector>& vector_set
) {
    double min_dot_prod = 1.0;
    for (const geom_vector& n : vector_set) {
        min_dot_prod = std::min(min_dot_prod, dot_product(d, n));
    }
    return min_dot_prod;
}

/**
 * @brief Compute the direction that minimises the maximal angle to a set of
 *        unit vectors.
 *
 * This function computes the unit vector d that minimises the largest angle
 * to the unit vectors n_k. This direction is the centre of the smallest
 * spherical cap that contains all n_k, which is defined by either one, two
 * or three of the n_k. The solution is found exactly by enumerating all
 * these candidate directions:
 * - single vectors: n_i
 * - pairs of vectors: normalised bisector of n_i and n_j
 * - triples of vectors: direction with equal angle to n_i, n_j and n_k
 * and selecting the candidate with the largest min_k(dot(d, n_k)). The
 * solution is only meaningful if all n_k lie in an open hemisphere
 * (min_cos > 0).
 *
 * @param vector_set Set of unit vectors [-].
 * @param min_cos Cosine of the largest angle between the computed direction
 *                and the unit vectors (corresponds to the minimal distance
 *                between the tip of the cap centre and the planes through
 *                the origin that have the unit vectors as normals) [-].
 * @return Cap centre vector (unit vector) [-].
 */
geom_vector smallest_enclosing_cap(
    const std::vector<geom_vector>& vector_set,
    double& min_cos
) {

    const std::vector<geom_vector>& n = vector_set;
    size_t num_vector = n.size();
    geom_vector cap_centre = {0.0, 0.0, 0.0};
    min_cos = -2.0;

    // Evaluate candidate direction and keep it if it is the best so far
    auto evaluate = [&](geom_vector d) {
        double d_mag = sqrt(dot_product(d, d));
        if (d_mag < 1.0e-12) {
            return;  // undefined direction (e.g. collinear vectors)
        }
        d = d * (1.0 / d_mag);
        double cos_d = min_dot_product(d, n);
        if (cos_d > min_cos) {
            min_cos = cos_d;
            cap_centre = d;
        }
    };

    for (size_t i = 0; i < num_vector; i++) {
        // Single vector
        evaluate(n[i]);
        for (size_t j = i + 1; j < num_vector; j++) {
            // Pair of vectors (bisector)
            evaluate({n[i].x + n[j].x, n[i].y + n[j].y, n[i].z + n[j].z});
            for (size_t k = j + 1; k < num_vector; k++) {
                // Triple of vectors (direction with equal angle to all three
                // vectors; orthogonal to the plane through their tips)
                geom_vector d = cross_product(n[j] - n[i], n[k] - n[i]);
                if (dot_product(d, n[i]) < 0.0) {
                    d = d * -1.0;
                }
                evaluate(d);
            }
        }
    }

    return cap_centre;

}

/**
 * @brief Compute the relative distance between the tip of 'dir_offset'
 *        (placed at the vertex) and the triangle, i.e. the distance divided
 *        by the offset length (1).
 *
 * First, it is checked if the perpendicular projection of the tip onto the
 * triangle plane lies within the wedge defined by the two edges of the
 * triangle at the vertex. If so, the distance to the plane is returned.
 * Otherwise, the distance to the closest edge or to the vertex is returned.
 *
 * @param dir_offset Offset direction (unit vector) [-].
 * @param tri_normal Normal of triangle (unit vector) [-].
 * @param edge_0 Direction of the first triangle edge at the vertex
 *               (unit vector) [-].
 * @param edge_1 Direction of the second triangle edge at the vertex
 *               (unit vector) [-].
 * @return Relative distance from the tip of 'dir_offset' to the wedge in the
 *         range [0.0, 1.0] [-].
 */
inline double wedge_clearance(
    geom_vector dir_offset,
    geom_vector tri_normal,
    geom_vector edge_0,
    geom_vector edge_1
) {

    // Check if perpendicular foot lies inside wedge defined by
    // q = a * edge_0 + b * edge_1; with a, b >= 0
    double dot_dn = dot_product(dir_offset, tri_normal);
    geom_vector q = dir_offset - tri_normal * dot_dn;
    geom_vector c = cross_product(edge_0, edge_1);
    double a = dot_product(cross_product(q, edge_1), c);
    double b = dot_product(cross_product(edge_0, q), c);
    if ((a >= 0.0) && (b >= 0.0)) {
        return std::fabs(dot_dn);
    }

    // Distance to closest edge or to vertex
    double clearance = 1.0;
    for (geom_vector e : {edge_0, edge_1}) {
        double dot_de = dot_product(dir_offset, e);
        if (dot_de > 0.0) {
            clearance = std::min(clearance,
                sqrt(std::max(0.0, 1.0 - dot_de * dot_de)));
        }
    }
    return clearance;

}

/**
 * @brief Compute the unit normals and the unit edge directions of the adjacent
 *        triangles of a vertex.
 *
 * The results are stored in 'adj.tri_normal', 'adj.tri_edge_0' and
 * 'adj.tri_edge_1'. The direction of the normal vector depends on the
 * winding order of the vertices. The edges point from the vertex to the two
 * other corners of the triangle.
 *
 * @param tri_vert Coordinates (x, y, z) of triangle vertices [m].
 * @param adj Adjacent triangles of the vertex with adj.idx_tri set by
 *            function 'find_adjacent_triangles'.
 */
void adjacent_triangle_geometry(
    const float* tri_vert,
    AdjacentTriangles& adj
){

    unsigned int idx_vert = adj.idx_vert;
    adj.tri_normal.clear();
    adj.tri_edge_0.clear();
    adj.tri_edge_1.clear();
    for (unsigned int idx_tri : adj.idx_tri) {
        const unsigned int* face = adj.tri_face + ((size_t)idx_tri * 3);
        geom_point vert[3];
        for (int j = 0; j < 3; j++) {
            vert[j] = {
                tri_vert[(face[j] * 3) + 0],
                tri_vert[(face[j] * 3) + 1],
                tri_vert[(face[j] * 3) + 2]
            };
        }
        adj.tri_normal.push_back(get_triangle_normal(vert[0], vert[1],
            vert[2]));
        int j_0 = (face[0] == idx_vert) ? 0 : ((face[1] == idx_vert) ? 1 : 2);
        geom_vector edge_0 = vert[(j_0 + 1) % 3] - vert[j_0];
        geom_vector edge_1 = vert[(j_0 + 2) % 3] - vert[j_0];
        unit_vector(edge_0);
        unit_vector(edge_1);
        adj.tri_edge_0.push_back(edge_0);
        adj.tri_edge_1.push_back(edge_1);
    }

}

/**
 * @brief Compute the optimal direction for offsetting the ray origin from a
 *        vertex and its clearance from the adjacent triangles.
 *
 * The offset direction is the unit vector that maximises the minimal distance
 * to the planes of the adjacent triangles (see function
 * 'smallest_enclosing_cap'). For a terrain (height field) mesh, this distance
 * is always positive, i.e. an offset along this direction lies above all
 * adjacent triangles. The mesh must not contain degenerated triangles and all
 * triangles must have an anti-clockwise winding (viewed from above).
 *
 * The clearance is the ratio between the minimal distance of the offset point
 * to the adjacent triangles (see function 'wedge_clearance') and the offset
 * length (1). An offset t [m] results in a minimal distance of
 * t * clearance [m]. At convex locations (e.g. peaks and ridges), it exceeds
 * the minimal distance to the planes.
 *
 * @param adj Adjacent triangles of the vertex with normals and edges (see
 *            function 'adjacent_triangle_geometry').
 * @param dir_offset Offset direction (unit vector) [-].
 * @param clearance Minimal relative distance from the tip of 'dir_offset'
 *                  to the adjacent triangles in the range (0.0, 1.0] [-].
 * @return Status if a valid offset direction was found.
 */
bool optimal_offset(
    const AdjacentTriangles& adj,
    geom_vector& dir_offset,
    double& clearance
){

    // Compute direction with maximal clearance from the triangle planes
    double min_cos;  // minimal distance to adjacent planes (per unit offset)
    dir_offset = smallest_enclosing_cap(adj.tri_normal, min_cos);
    if (min_cos <= 0.0) {
        return false;  // no common open hemisphere exists
    }

    // Compute exact clearance from adjacent triangles
    clearance = 1.0;  // clearance >= min_cos
    for (size_t i = 0; i < adj.tri_normal.size(); i++) {
        clearance = std::min(clearance, wedge_clearance(dir_offset,
            adj.tri_normal[i], adj.tri_edge_0[i], adj.tri_edge_1[i]));
    }
    return true;

}

//-----------------------------------------------------------------------------
// Embree related functions
//-----------------------------------------------------------------------------

/**
 * @brief Error function for device initialiser.
 * @param userPtr
 * @param error
 * @param str
 */
void errorFunction(void* userPtr, enum RTCError error, const char* str) {
	printf("error %d: %s\n", error, str);
}

/**
 * @brief Initialises device and registers error handler
 * @return Device instance.
 */
RTCDevice initializeDevice() {
	RTCDevice device = rtcNewDevice(NULL);
  	if (!device) {
    	printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));
    }
  	rtcSetDeviceErrorFunction(device, errorFunction, NULL);
  	return device;
}

/**
 * @brief Initialises the Embree scene.
 * @param device Initialised device.
 * @param tri_vert Coordinates (x, y, z) of triangle vertices [m].
 * @param tri_face Faces of triangle mesh.
 * @param num_vert Number of triangle vertices.
 * @param num_face Number of triangles (faces).
 * @return Embree scene.
 */
RTCScene initializeScene(RTCDevice
    device,
    float* tri_vert,
    unsigned int* tri_face,
    int num_vert,
    int num_face
){

    RTCScene scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Triangle vertices
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX,
        0, RTC_FORMAT_FLOAT3, tri_vert, 0, 3 * sizeof(float), num_vert);

    // Triangle faces
    rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
        RTC_FORMAT_UINT3, tri_face, 0, 3 * sizeof(unsigned int), num_face);

    auto start = std::chrono::high_resolution_clock::now();

    // Commit geometry and scene
    rtcCommitGeometry(geom);
    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);
    rtcCommitScene(scene);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "Building bounding volume hierarchy (BVH): " << time.count()
        << " s" << std::endl;

    return scene;

}

/**
 * @brief Ray casting with occlusion testing (hit / no hit).
 * @param scene Embree scene.
 * @param ox x-coordinate of the ray origin [m].
 * @param oy y-coordinate of the ray origin [m].
 * @param oz z-coordinate of the ray origin [m].
 * @param dx x-component of the ray direction [m].
 * @param dy y-component of the ray direction [m].
 * @param dz z-component of the ray direction [m].
 * @param dist_search Search distance for potential collision [m].
 * @return Collision status (true: hit, false: no hit).
 */
bool castRay_occluded1(RTCScene scene, float ox, float oy, float oz, float dx,
    float dy, float dz, float dist_search){
    struct RTCRay ray;
    ray.org_x = ox;
    ray.org_y = oy;
    ray.org_z = oz;
    ray.dir_x = dx;
    ray.dir_y = dy;
    ray.dir_z = dz;
    ray.tnear = 0.0;
    ray.tfar = dist_search;
    ray.mask = 1;
    rtcOccluded1(scene, &ray); // intersect ray with scene
    return (ray.tfar < 0.0);
}

/**
 * @brief Ray casting with intersection testing (distance to closest hit).
 * @param scene Embree scene.
 * @param ox x-coordinate of the ray origin [m].
 * @param oy y-coordinate of the ray origin [m].
 * @param oz z-coordinate of the ray origin [m].
 * @param dx x-component of the ray direction (unit vector) [-].
 * @param dy y-component of the ray direction (unit vector) [-].
 * @param dz z-component of the ray direction (unit vector) [-].
 * @param dist_search Search distance for potential collision [m].
 * @return Distance to the closest hit (negative value: no hit) [m].
 */
float castRay_intersect1(RTCScene scene, float ox, float oy, float oz,
    float dx, float dy, float dz, float dist_search){
    struct RTCRayHit rayhit;
    rayhit.ray.org_x = ox;
    rayhit.ray.org_y = oy;
    rayhit.ray.org_z = oz;
    rayhit.ray.dir_x = dx;
    rayhit.ray.dir_y = dy;
    rayhit.ray.dir_z = dz;
    rayhit.ray.tnear = 0.0f;
    rayhit.ray.tfar = dist_search;
    rayhit.ray.time = 0.0f;
    rayhit.ray.mask = 1;
    rayhit.ray.flags = 0;
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
    rtcIntersect1(scene, &rayhit); // intersect ray with scene
    if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
        return -1.0f;
    }
    return rayhit.ray.tfar;
}

/**
 * @brief Computes the terrain horizon for a specific point.
 *
 * This function computes the terrain horizon for a specific point on the
 * triangle mesh. It iteratively samples a certain azimuth direction with rays
 * until the horizon is found. For all but the first azimuth direction, the
 * elevation angle for the search is initialised with a value equal to the
 * horizon from the previous azimuth direction +/- the horizon accuracy value.
 *
 * @param scene Embree scene.
 * @param ray_origin ray origin [m].
 * @param sphere_normal Sphere normal at the point location [m].
 * @param north_dir North direction at the point location [m].
 * @param num_azim Number of azimuth directions.
 * @param azim_offset Offset of first azimuth position from 0.0 [rad].
 * @param horizon_acc Horizon accuracy [rad].
 * @param dist_search Search distance for potential collision [m].
 * @param elev_angle_min Threshold angle for sampling in negative elevation
 *                       angle direction [rad].
 * @param azim_sin Sine of the azimuth angle spacing.
 * @param azim_cos Cosine of the azimuth angle spacing.
 * @param elev_sin_2ha Sine of the double elevation angle spacing.
 * @param elev_cos_2ha Cosine of the double elevation angle spacing.
 * @param horizon Horizon array [rad].
 * @param idx_horizon Index for horizon array
 * @param num_rays Number of rays cast.
 */
void terrain_horizon(
    RTCScene scene,
    geom_point ray_origin,
    geom_vector sphere_normal,
    geom_vector north_dir,
    int num_azim,
    double azim_offset,
    double horizon_acc,
    float dist_search,
    double elev_angle_min,
    double azim_sin,
    double azim_cos,
    double elev_sin_2ha,
    double elev_cos_2ha,
    float* horizon,
    size_t idx_horizon,
    size_t &num_rays
){

    // Initial ray direction
    geom_vector ray_dir;
    ray_dir.x = north_dir.x;
    ray_dir.y = north_dir.y;
    ray_dir.z = north_dir.z;

    // Ray origin
    float ray_origin_x = (float)ray_origin.x;
    float ray_origin_y = (float)ray_origin.y;
    float ray_origin_z = (float)ray_origin.z;

    // Shift azimuth angle in case of 'refine_factor' > 1 so that first
    // azimuth sector is centred around 0.0 deg (North)
    ray_dir = vector_rotation(ray_dir, sphere_normal, sin(-azim_offset),
        cos(-azim_offset));

    // Sample along azimuth
    double elev_angle = 0.0;
    for (int i = 0; i < num_azim; i++){

        // Rotation axis
        geom_vector rot_axis = cross_product(ray_dir, sphere_normal);
        unit_vector(rot_axis);
        // not necessarily a unit vector because vectors are mostly not
        // perpendicular

        // Find terrain horizon by iterative ray sampling
        bool hit = castRay_occluded1(scene, ray_origin_x, ray_origin_y,
            ray_origin_z, (float)ray_dir.x, (float)ray_dir.y, (float)ray_dir.z,
            dist_search);
        num_rays += 1;
        if (hit) { // terrain hit -> increase elevation angle
            while (hit){
                elev_angle += (2.0 * horizon_acc);
                ray_dir = vector_rotation(ray_dir, rot_axis, elev_sin_2ha,
                    elev_cos_2ha);
                hit = castRay_occluded1(scene, ray_origin_x, ray_origin_y,
                ray_origin_z, (float)ray_dir.x, (float)ray_dir.y,
                (float)ray_dir.z, dist_search);
                num_rays += 1;
            }
            horizon[idx_horizon * num_azim + i] = elev_angle - horizon_acc;
        } else { // terrain not hit -> decrease elevation angle
            while ((!hit) && (elev_angle > elev_angle_min)){
                elev_angle -= (2.0 * horizon_acc);
                ray_dir = vector_rotation(ray_dir, rot_axis, -elev_sin_2ha,
                    elev_cos_2ha); // sin(-x) == -sin(x), cos(x) == cos(-x)
                hit = castRay_occluded1(scene, ray_origin_x, ray_origin_y,
                ray_origin_z, (float)ray_dir.x, (float)ray_dir.y,
                (float)ray_dir.z, dist_search);
                num_rays += 1;
            }
            horizon[idx_horizon * num_azim + i] = elev_angle + horizon_acc;
        }

        // Azimuthal rotation of ray direction (clockwise; first to east)
        ray_dir = vector_rotation(ray_dir, sphere_normal, -azim_sin,
            azim_cos);  // sin(-x) == -sin(x), cos(x) == cos(-x)

    }

}

/**
 * @brief Callback function for the point query that searches the adjacent
 *        triangles of a vertex.
 *
 * Embree calls this function for every triangle in the BVH leaf nodes that
 * overlap with the query domain. Only triangles that actually contain the
 * query vertex (according to 'tri_face') are kept.
 *
 * @param args Point query function arguments.
 * @return Status if query radius was modified (always false).
 */
bool adjacent_triangles_callback(RTCPointQueryFunctionArguments* args) {
    AdjacentTriangles* adj = (AdjacentTriangles*)args->userPtr;
    const unsigned int* face = adj->tri_face + ((size_t)args->primID * 3);
    if ((face[0] == adj->idx_vert) || (face[1] == adj->idx_vert)
        || (face[2] == adj->idx_vert)) {
        adj->idx_tri.push_back(args->primID);
    }
    return false;
}

/**
 * @brief Search the adjacent triangles of a vertex.
 *
 * The adjacent triangles of the vertex are searched with an Embree point
 * query. Their indices are stored in 'adj.idx_tri'.
 *
 * @param scene Embree scene.
 * @param tri_vert Coordinates (x, y, z) of triangle vertices [m].
 * @param idx_vert Index of the vertex.
 * @param query_radius Radius of the point query [m].
 * @param adj User data for the point query (reused between calls).
 * @return Status if at least one adjacent triangle was found.
 */
bool find_adjacent_triangles(
    RTCScene scene,
    const float* tri_vert,
    unsigned int idx_vert,
    float query_radius,
    AdjacentTriangles& adj
){

    adj.idx_vert = idx_vert;
    adj.idx_tri.clear();
    struct RTCPointQuery query;
    query.x = tri_vert[(idx_vert * 3) + 0];
    query.y = tri_vert[(idx_vert * 3) + 1];
    query.z = tri_vert[(idx_vert * 3) + 2];
    query.time = 0.0f;
    query.radius = query_radius;
    struct RTCPointQueryContext context;
    rtcInitPointQueryContext(&context);
    rtcPointQuery(scene, &query, &context, adjacent_triangles_callback,
        (void*)&adj);

    // Remove duplicates (a triangle can be referenced by multiple leaf nodes)
    std::sort(adj.idx_tri.begin(), adj.idx_tri.end());
    adj.idx_tri.erase(std::unique(adj.idx_tri.begin(), adj.idx_tri.end()),
        adj.idx_tri.end());
    return !adj.idx_tri.empty();

}

//-----------------------------------------------------------------------------
// Class constructor/destructor and class methods
//-----------------------------------------------------------------------------

CppTerrain::CppTerrain() {

    device = initializeDevice();
    scene = nullptr;
}

CppTerrain::~CppTerrain() {

  	// Release resources allocated through Embree
    if (scene) {
        rtcReleaseScene(scene);
    }
    if (device) {
        rtcReleaseDevice(device);
    }
}

void CppTerrain::initialise(
    float* tri_vert,
    unsigned int* tri_face,
    int num_vert,
    int num_face,
    float* earth_centre,
    float* north_pole
) {

    tri_vert_cl = tri_vert;
    tri_face_cl = tri_face;
    num_vert_cl = num_vert;
    num_face_cl = num_face;
    earth_centre_cl = {earth_centre[0], earth_centre[1], earth_centre[2]};
    north_pole_cl = {north_pole[0], north_pole[1], north_pole[2]};

    cout << "Number of triangle mesh vertices: " << num_vert << endl;
    cout << "Number of triangle mesh faces: " << num_face << endl;

    // Maximal absolute coordinate of triangle mesh vertices
    max_abs_coord_cl = 0.0;
    for (size_t i = 0; i < (size_t)num_vert * 3; i++) {
        max_abs_coord_cl = std::max(max_abs_coord_cl,
            (double)std::fabs(tri_vert[i]));
    }
    cout << std::setprecision(1) << std::fixed
        << "Maximal absolute coordinate of triangle mesh: "
        << max_abs_coord_cl << " m" << endl;

    // Check that all triangles are non-degenerated (non-zero area) and have
    // an anti-clockwise winding (viewed from above), i.e. that their normals
    // point away from the Earth's centre
    std::atomic<size_t> num_degenerated(0);
    std::atomic<size_t> num_clockwise(0);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, (size_t)num_face),
    [&](tbb::blocked_range<size_t> r) {
    for (size_t i=r.begin(); i<r.end(); ++i) {
        geom_point vert[3];
        for (int j = 0; j < 3; j++) {
            unsigned int idx = tri_face[(i * 3) + j];
            vert[j] = {
                tri_vert[(idx * 3) + 0],
                tri_vert[(idx * 3) + 1],
                tri_vert[(idx * 3) + 2]
            };
        }
        geom_vector tri_normal = cross_product(vert[2] - vert[1],
            vert[0] - vert[1]);  // same orientation as 'get_triangle_normal'
        geom_vector radial = compute_centroid(vert[0], vert[1], vert[2])
            - earth_centre_cl;
        if (dot_product(tri_normal, tri_normal) == 0.0) {
            num_degenerated++;
        } else if (dot_product(tri_normal, radial) <= 0.0) {
            num_clockwise++;
        }
    }
    });
    if ((num_degenerated > 0) || (num_clockwise > 0)) {
        std::ostringstream msg;
        msg << "Invalid triangle mesh: " << num_degenerated
            << " degenerated triangle(s) (zero area) and " << num_clockwise
            << " triangle(s) without anti-clockwise winding (normal does not "
            "point away from the Earth's centre)";
        throw std::invalid_argument(msg.str());
    }

    // Build bounding volume hierarchy (BVH)
    scene = initializeScene(
        device,
        tri_vert,
        tri_face,
        num_vert,
        num_face
    );

}

void CppTerrain::horizon_vertex(
    int num_azim,
    double azim_offset,
    unsigned int* slice_loc,
    float dist_search,
    double ray_offset_factor,
    double horizon_acc,
    double elev_angle_min,
    float* horizon,
    float* clearance_out
) {

    // Convert units
    azim_offset = deg2rad(azim_offset);
    horizon_acc = deg2rad(horizon_acc);
    elev_angle_min = deg2rad(elev_angle_min);

    // Compute offset of ray origin from vertex
    double ray_offset = ray_offset_factor * (double)FLT_EPSILON
        * max_abs_coord_cl;
    cout << std::setprecision(3) << std::fixed
        << "Offset of ray origin (unadjusted): " << ray_offset << " m" << endl;

    // Radius of point query to search adjacent triangles of vertex
    float query_radius = (float)(10.0 * (double)FLT_EPSILON
        * max_abs_coord_cl);
    // (triangles that are found but not adjacent are removed via 'tri_face')

    // Lower limit for clearance from adjacent triangles (-> offset is
    // increased by at most a factor of 10)
    const double clearance_lim = 0.1;  // ca. 84.3 deg

    std::atomic<size_t> num_fallback(0);

    // Evaluated trigonometric functions for rotation along azimuth/elevation
    // angle
    double azim_sin = sin(deg2rad(360.0) / (double)num_azim);
    double azim_cos = cos(deg2rad(360.0) / (double)num_azim);
    double elev_sin_2ha = sin(2.0 * horizon_acc);
    double elev_cos_2ha = cos(2.0 * horizon_acc);
    // Note: sin(-x) == -sin(x), cos(x) == cos(-x)

    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
    tbb::blocked_range<size_t>(slice_loc[0], slice_loc[1]), 0.0,
    [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

    // User data for point query (private for each TBB task)
    AdjacentTriangles adj;
    adj.tri_face = tri_face_cl;
    adj.idx_tri.reserve(16);
    adj.tri_normal.reserve(16);
    adj.tri_edge_0.reserve(16);
    adj.tri_edge_1.reserve(16);

    // for (size_t i = (size_t)slice_loc[0]; i < (size_t)slice_loc[1]; i++){
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

        // Compute sphere normal
        geom_point vertex = {
            tri_vert_cl[(i * 3) + 0],
            tri_vert_cl[(i * 3) + 1],
            tri_vert_cl[(i * 3) + 2]
        };
        geom_vector sphere_normal = vertex - earth_centre_cl;
        unit_vector(sphere_normal);

        // Compute north direction (orthogonal to sphere normal)
        geom_vector v_n = north_pole_cl - vertex;
        double dot_prod = dot_product(v_n, sphere_normal);
        geom_vector north_dir = v_n - sphere_normal * dot_prod;
        unit_vector(north_dir);

        // Compute optimal offset direction and clearance
        geom_vector dir_offset;
        double clearance;
        bool valid = find_adjacent_triangles(scene, tri_vert_cl,
            (unsigned int)i, query_radius, adj);
        if (valid) {
            adjacent_triangle_geometry(tri_vert_cl, adj);
            valid = optimal_offset(adj, dir_offset, clearance);
        }
        if (!valid) {
            dir_offset = sphere_normal;
            clearance = 1.0;
            num_fallback++;
        }

        // Save clearance (before clipping; NaN for no valid offset direction)
        if (clearance_out) {
            clearance_out[i - slice_loc[0]] = valid ? (float)clearance
                : std::numeric_limits<float>::quiet_NaN();
        }

        clearance = std::max(clearance, clearance_lim);

        // Offset ray origin along an optimal direction (scaled so that
        // the distance to all adjacent triangles is at least 'ray_offset')
        geom_point ray_origin = vertex + dir_offset * (ray_offset / clearance);
        // The origin of the ray is slightly offset to avoid potential ray-
        // terrain collisions near the origin due to numerical imprecisions.

        // Compute terrain horizon
        terrain_horizon(
            scene,
            ray_origin,
            sphere_normal,
            north_dir,
            num_azim,
            azim_offset,
            horizon_acc,
            dist_search,
            elev_angle_min,
            azim_sin,
            azim_cos,
            elev_sin_2ha,
            elev_cos_2ha,
            horizon,
            i - slice_loc[0],
            num_rays
        );

    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = end_ray - start_ray;
    cout << std::setprecision(2) << "Ray tracing: " << time_ray.count()
        << " s" << endl;

    // Print number of rays needed for location and azimuth direction
    cout << "Number of rays shot: " << num_rays << std::endl;
    double ratio = (double)num_rays
        / (double)((slice_loc[1] - slice_loc[0]) * num_azim);
    cout << std::setprecision(2) << "Average number of rays per cell and "
        "azimuth sector: " << ratio << endl;

    // Print number of vertices with sphere normal as fallback offset direction
    if (num_fallback > 0) {
        cout << "Warning: no valid offset direction found for "
            << num_fallback << " vertices (sphere normal used instead)"
            << endl;
    }

}

void CppTerrain::horizon_centroid(
    int num_azim,
    unsigned int* slice_loc,
    float dist_search,
    double ray_offset_factor,
    double horizon_acc,
    double elev_angle_min,
    float* horizon
) {

    // Convert units
    horizon_acc = deg2rad(horizon_acc);
    elev_angle_min = deg2rad(elev_angle_min);

    // Compute offset of ray origin from triangle surface
    double ray_offset = ray_offset_factor * (double)FLT_EPSILON
        * max_abs_coord_cl;
    cout << std::setprecision(3) << std::fixed
        << "Offset of ray origin: " << ray_offset << " m" << endl;

    // Evaluated trigonometric functions for rotation along azimuth/elevation
    // angle
    double azim_sin = sin(deg2rad(360.0) / (double)num_azim);
    double azim_cos = cos(deg2rad(360.0) / (double)num_azim);
    double elev_sin_2ha = sin(2.0 * horizon_acc);
    double elev_cos_2ha = cos(2.0 * horizon_acc);
    // Note: sin(-x) == -sin(x), cos(x) == cos(-x)

    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
    tbb::blocked_range<size_t>(slice_loc[0], slice_loc[1]), 0.0,
    [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

    // for (size_t i = (size_t)slice_loc[0]; i < (size_t)slice_loc[1]; i++){
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

        // Get vertices of triangle
        unsigned int idx_0 = tri_face_cl[(i * 3) + 0];
        geom_point vert_0 = {
            tri_vert_cl[(idx_0 * 3) + 0],
            tri_vert_cl[(idx_0 * 3) + 1],
            tri_vert_cl[(idx_0 * 3) + 2]
        };
        unsigned int idx_1 = tri_face_cl[(i * 3) + 1];
        geom_point vert_1 = {
            tri_vert_cl[(idx_1 * 3) + 0],
            tri_vert_cl[(idx_1 * 3) + 1],
            tri_vert_cl[(idx_1 * 3) + 2]
        };
        unsigned int idx_2 = tri_face_cl[(i * 3) + 2];
        geom_point vert_2 = {
            tri_vert_cl[(idx_2 * 3) + 0],
            tri_vert_cl[(idx_2 * 3) + 1],
            tri_vert_cl[(idx_2 * 3) + 2]
        };

        // Compute triangle centroid and triangle normal (unit vector)
        geom_point centroid = compute_centroid(vert_0, vert_1, vert_2);
        geom_vector tri_normal = get_triangle_normal(vert_0, vert_1, vert_2);

        // Compute sphere normal and north direction (orthogonal to sphere
        // normal)
        geom_vector sphere_normal = centroid - earth_centre_cl;
        unit_vector(sphere_normal);
        geom_vector v_n = north_pole_cl - centroid;
        double dot_prod = dot_product(v_n, sphere_normal);
        geom_vector north_dir = v_n - sphere_normal * dot_prod;
        unit_vector(north_dir);

        // Offset ray origin along the triangle normal
        geom_point ray_origin = centroid + tri_normal * ray_offset;
        // The origin of the ray is slightly offset to avoid potential ray-
        // terrain collisions near the origin due to numerical imprecisions.

        // Compute terrain horizon
        terrain_horizon(
            scene,
            ray_origin,
            sphere_normal,
            north_dir,
            num_azim,
            0.0,
            horizon_acc,
            dist_search,
            elev_angle_min,
            azim_sin,
            azim_cos,
            elev_sin_2ha,
            elev_cos_2ha,
            horizon,
            i - slice_loc[0],
            num_rays
        );

    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = end_ray - start_ray;
    cout << std::setprecision(2) << "Ray tracing: " << time_ray.count()
        << " s" << endl;

    // Print number of rays needed for location and azimuth direction
    cout << "Number of rays shot: " << num_rays << std::endl;
    double ratio = (double)num_rays
        / (double)((slice_loc[1] - slice_loc[0]) * num_azim);
    cout << std::setprecision(2) << "Average number of rays per cell and "
        "azimuth sector: " << ratio << endl;

}

void CppTerrain::horizon_arbitrary(
    double* coords,
    int num_loc,
    int num_azim,
    float dist_search,
    double height_min,
    double horizon_acc,
    double elev_angle_min,
    float* horizon,
    float* height_above
) {

    // Convert units
    horizon_acc = deg2rad(horizon_acc);
    elev_angle_min = deg2rad(elev_angle_min);

    // Height above the location from which a ray is cast downwards to
    // determine the height of the location above the triangle mesh (must
    // exceed the terrain height above the location)
    const double ray_start_height = 10000.0;  // [m]

    std::atomic<size_t> num_outside(0);

    // Evaluated trigonometric functions for rotation along azimuth/elevation
    // angle
    double azim_sin = sin(deg2rad(360.0) / (double)num_azim);
    double azim_cos = cos(deg2rad(360.0) / (double)num_azim);
    double elev_sin_2ha = sin(2.0 * horizon_acc);
    double elev_cos_2ha = cos(2.0 * horizon_acc);
    // Note: sin(-x) == -sin(x), cos(x) == cos(-x)

    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, (size_t)num_loc), (size_t)0,
    [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

    // for (size_t i = 0; i < (size_t)num_loc; i++){
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

        // Compute sphere normal
        geom_point loc = {
            coords[(i * 3) + 0],
            coords[(i * 3) + 1],
            coords[(i * 3) + 2]
        };
        geom_vector sphere_normal = loc - earth_centre_cl;
        unit_vector(sphere_normal);

        // Compute north direction (orthogonal to sphere normal)
        geom_vector v_n = north_pole_cl - loc;
        double dot_prod = dot_product(v_n, sphere_normal);
        geom_vector north_dir = v_n - sphere_normal * dot_prod;
        unit_vector(north_dir);

        // Compute height of location above the triangle mesh
        geom_point ray_start = loc + sphere_normal * ray_start_height;
        float dist = castRay_intersect1(scene, (float)ray_start.x,
            (float)ray_start.y, (float)ray_start.z, (float)-sphere_normal.x,
            (float)-sphere_normal.y, (float)-sphere_normal.z,
            std::numeric_limits<float>::infinity());
        if (dist < 0.0f) {
            // location outside of triangle mesh
            height_above[i] = std::numeric_limits<float>::quiet_NaN();
            for (int j = 0; j < num_azim; j++) {
                horizon[(i * num_azim) + j]
                    = std::numeric_limits<float>::quiet_NaN();
            }
            num_outside++;
            continue;
        }
        double height = (double)dist - ray_start_height;
        height_above[i] = (float)height;

        // Elevate location to minimal height above triangle mesh (if
        // required)
        geom_point ray_origin = loc;
        if (height < height_min) {
            ray_origin = loc + sphere_normal * (height_min - height);
        }

        // Compute terrain horizon
        terrain_horizon(
            scene,
            ray_origin,
            sphere_normal,
            north_dir,
            num_azim,
            0.0,
            horizon_acc,
            dist_search,
            elev_angle_min,
            azim_sin,
            azim_cos,
            elev_sin_2ha,
            elev_cos_2ha,
            horizon,
            i,
            num_rays
        );

    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = end_ray - start_ray;
    cout << std::setprecision(2) << "Ray tracing: " << time_ray.count()
        << " s" << endl;

    // Print number of rays needed for location and azimuth direction
    cout << "Number of rays shot: " << num_rays << std::endl;
    size_t num_loc_inside = (size_t)num_loc - num_outside;
    if (num_loc_inside > 0) {
        double ratio = (double)num_rays
            / (double)(num_loc_inside * (size_t)num_azim);
        cout << std::setprecision(2) << "Average number of rays per location "
            "and azimuth sector: " << ratio << endl;
    }

    // Print number of locations outside of triangle mesh
    if (num_outside > 0) {
        cout << "Warning: " << num_outside << " locations outside of triangle "
            "mesh" << endl;
    }

}
