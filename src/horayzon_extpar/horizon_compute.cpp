#include "horizon_compute.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>

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
// Functions (not dependent on Embree)
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
 * @param k Unit vector specifying the rotation axis.
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
 * This function computes the nomal (unit vector) of a triangle. The direction
 * of the normal vector depends on the ordering of the vertices.
 *
 * @param vert_0 Vertex 0.
 * @param vert_1 Vertex 1.
 * @param vert_2 Vertex 2.
 * @return Triangle normal (unit vector).
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

//-----------------------------------------------------------------------------
// Functions (Embree related)
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
 * @param num_rays Number of rays casted.
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
    double ray_origin_elev,
    double horizon_acc,
    double elev_angle_min,
    float* horizon
) {

    // Convert units
    azim_offset = deg2rad(azim_offset);
    horizon_acc = deg2rad(horizon_acc);
    elev_angle_min = deg2rad(elev_angle_min);

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

        // Elevate origin for ray tracing by 'safety margin'
        geom_point ray_origin = vertex + sphere_normal * ray_origin_elev;
        // The origin of the ray is slightly elevated to avoid potential ray-
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

}

void CppTerrain::horizon_centroid(
    int num_azim,
    unsigned int* slice_loc,
    float dist_search,
    double ray_origin_elev,
    double horizon_acc,
    double elev_angle_min,
    float* horizon
) {

    // Convert units
    horizon_acc = deg2rad(horizon_acc);
    elev_angle_min = deg2rad(elev_angle_min);

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

        // Elevate origin for ray tracing by 'safety margin'
        // geom_point ray_origin = centroid + sphere_normal * ray_origin_elev;
        geom_point ray_origin = centroid + tri_normal * ray_origin_elev;
        // The origin of the ray is slightly elevated to avoid potential ray-
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