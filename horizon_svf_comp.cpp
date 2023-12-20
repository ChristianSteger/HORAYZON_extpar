#define _USE_MATH_DEFINES
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <exception>
#include <math.h>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <embree3/rtcore.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

using namespace std;

//#############################################################################
// Geometries
//#############################################################################

struct geom_point{
    double x, y, z;
};

struct geom_vector{
    double x, y, z;
};

//#############################################################################
// Functions
//#############################################################################

// ----------------------------------------------------------------------------
// Unit conversion
// ----------------------------------------------------------------------------

// Convert degree to radian
inline double deg2rad(double ang) {
	return ((ang / 180.0) * M_PI);
}

// Convert radian to degree
inline double rad2deg(double ang) {
	return ((ang / M_PI) * 180.0);
}

// ----------------------------------------------------------------------------
// Geometrical operations
// ----------------------------------------------------------------------------

// Dot product
inline double dot_product(geom_vector a, geom_vector b) {
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

// Cross product
inline geom_vector cross_product(geom_vector a, geom_vector b) {
    geom_vector c = {a.y * b.z - a.z * b.y,
                     a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x};
    return c;
}

// Unit vector
void unit_vector(geom_vector& a) {
    double vector_mag = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    a.x /= vector_mag;
    a.y /= vector_mag;
    a.z /= vector_mag;
}

// Rodrigues' rotation formula
inline geom_vector vector_rotation(geom_vector v, geom_vector k,
    double ang_rot_sin, double ang_rot_cos) {
    geom_vector v_rot;
    double term = dot_product(k, v) * (1.0 - ang_rot_cos);
    v_rot.x = v.x * ang_rot_cos + (k.y * v.z - k.z * v.y) * ang_rot_sin
        + k.x * term;
    v_rot.y = v.y * ang_rot_cos + (k.z * v.x - k.x * v.z) * ang_rot_sin
        + k.y * term;
    v_rot.z = v.z * ang_rot_cos + (k.x * v.y - k.y * v.x) * ang_rot_sin
        + k.z * term;
    return v_rot;
}

// Multiply matrix and vector
inline geom_vector vector_matrix_multiplication(geom_vector v_in,
    double matrix[3][3]) {
    geom_vector v_out;
    v_out.x = matrix[0][0] * v_in.x + matrix[0][1] * v_in.y
        + matrix[0][2] * v_in.z;
    v_out.y = matrix[1][0] * v_in.x + matrix[1][1] * v_in.y
        + matrix[1][2] * v_in.z;
    v_out.z = matrix[2][0] * v_in.x + matrix[2][1] * v_in.y
        + matrix[2][2] * v_in.z;
    return v_out;
}

//#############################################################################
// Coordinate transformation and north vector
//#############################################################################

std::vector<geom_point> lonlat2ecef(double* lon, double* lat,
    float* elevation, int num_point, double rad_earth){

    /*Transformation of geographic longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates. Assume spherical Earth.

    Parameters
    ----------
    lon : array of double
        Array with geographic longitude [rad]
    lat : array of double
        Array with geographic latitude [rad]
    elevation : array of float
        Array with elevation above sphere [m]
    num_point : int
        Number of points
	rad_earth : double
	    Radius of Earth [m]

    Returns
    -------
    points : vector of type <geom_point>
		Vector of points (x, y, z) in ECEF coordinates [m]*/

    vector<geom_point> points(num_point);
    for (int i = 0; i < num_point; i++){
        points[i].x = (rad_earth + elevation[i]) * cos(lat[i]) * cos(lon[i]);
        points[i].y = (rad_earth + elevation[i]) * cos(lat[i]) * sin(lon[i]);
        points[i].z = (rad_earth + elevation[i]) * sin(lat[i]);
	}
    return points;
}

// ----------------------------------------------------------------------------

std::vector<geom_vector> north_direction(vector<geom_point> points,
    vector<geom_vector> sphere_normals, double rad_earth){

    /* Compute unit vectors for points in earth-centered, earth-fixed (ECEF)
    coordinates that point towards North and are perpendicular to the sphere's
    normals.

    Parameters
    ----------
    points : vector of type <geom_point>
		Vector of points (x, y, z) in ECEF coordinates [m]
    sphere_normals : vector of type <geom_vector>
		Vector of sphere normals (x, y, z) at the point locations in ECEF
		coordinates [m]
	rad_earth : double
	    Radius of Earth [m]

    Returns
    -------
    north_directions : vector of type <geom_vector>
		Vector with north directions (x, y, z) in ECEF coordinates [m]*/

    geom_vector v_p = {0.0, 0.0, rad_earth};  // north pole in ECEF coordinates
	vector<geom_vector> north_directions(points.size());
    geom_vector v_n, v_j;
    double dot_prod;
    for (size_t i = 0; i < points.size(); i++){
        v_n.x = v_p.x - points[i].x;
        v_n.y = v_p.y - points[i].y;
        v_n.z = v_p.z - points[i].z;
        dot_prod = dot_product(v_n, sphere_normals[i]);
        v_j.x = v_n.x - dot_prod * sphere_normals[i].x;
        v_j.y = v_n.y - dot_prod * sphere_normals[i].y;
        v_j.z = v_n.z - dot_prod * sphere_normals[i].z;
        unit_vector(v_j);
        north_directions[i] = v_j;
    }
    return north_directions;
}

// ----------------------------------------------------------------------------

void ecef2enu_point(vector<geom_point>& points, double lon_orig,
    double lat_orig, double rad_earth){

    /* In-place coordinate transformation of points from ECEF to ENU
    coordinate system.

    Parameters
    ----------
    points : vector of type <geom_point>
		Vector of points (x, y, z) in ECEF (in) coordinates [m]
    lon_orig : double
        Longitude of ENU coordinate system origin [rad]
    lat_orig : double
        Latitude of ENU coordinate system origin [rad]
	rad_earth : double
	    Radius of Earth [m]*/

    double sin_lon = sin(lon_orig);
    double cos_lon = cos(lon_orig);
    double sin_lat = sin(lat_orig);
    double cos_lat = cos(lat_orig);

    double x_ecef_orig = rad_earth * cos(lat_orig) * cos(lon_orig);
    double y_ecef_orig = rad_earth * cos(lat_orig) * sin(lon_orig);
    double z_ecef_orig = rad_earth * sin(lat_orig);

    double x_enu, y_enu, z_enu;
    for (size_t i = 0; i < points.size(); i++){
        x_enu = - sin_lon * (points[i].x - x_ecef_orig)
            + cos_lon * (points[i].y - y_ecef_orig);
        y_enu = - sin_lat * cos_lon * (points[i].x - x_ecef_orig)
            - sin_lat * sin_lon * (points[i].y - y_ecef_orig)
            + cos_lat * (points[i].z - z_ecef_orig);
        z_enu = + cos_lat * cos_lon * (points[i].x - x_ecef_orig)
            + cos_lat * sin_lon * (points[i].y - y_ecef_orig)
            + sin_lat * (points[i].z - z_ecef_orig);
        points[i].x = x_enu;
        points[i].y = y_enu;
        points[i].z = z_enu;
    }
}

// ----------------------------------------------------------------------------

void ecef2enu_vector(vector<geom_vector>& vectors, double lon_orig,
    double lat_orig){

    /*In-place coordinate transformation of vectors from ECEF to ENU
    coordinate system.

    Parameters
    ----------
    vectors : vector of type <geom_vector>
		Vector of vectors (x, y, z) in ECEF (in) coordinates [m]
    lon_orig : double
        Longitude of ENU coordinate system origin [rad]
    lat_orig : double
        Latitude of ENU coordinate system origin [rad]*/

    double sin_lon = sin(lon_orig);
    double cos_lon = cos(lon_orig);
    double sin_lat = sin(lat_orig);
    double cos_lat = cos(lat_orig);

    double x_enu, y_enu, z_enu;
    for (size_t i = 0; i < vectors.size(); i++){
        x_enu = - sin_lon * vectors[i].x
            + cos_lon * vectors[i].y;
        y_enu = - sin_lat * cos_lon * vectors[i].x
            - sin_lat * sin_lon * vectors[i].y
            + cos_lat * vectors[i].z;
        z_enu = + cos_lat * cos_lon * vectors[i].x
            + cos_lat * sin_lon * vectors[i].y
            + sin_lat * vectors[i].z;
        vectors[i].x = x_enu;
        vectors[i].y = y_enu;
        vectors[i].z = z_enu;
    }
}

//#############################################################################
// Miscellaneous
//#############################################################################

// Namespace
#if defined(RTC_NAMESPACE_USE)
    RTC_NAMESPACE_USE
#endif

// Error function
void errorFunction(void* userPtr, enum RTCError error, const char* str) {
    printf("error %d: %s\n", error, str);
}

// Initialisation of device and registration of error handler
RTCDevice initializeDevice() {
    RTCDevice device = rtcNewDevice(NULL);
    if (!device) {
        printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));
    }
    rtcSetDeviceErrorFunction(device, errorFunction, NULL);
    return device;
}

//#############################################################################
// Create scene from geometries
//#############################################################################

// Structures for vertices and triangle
struct Vertex{float x, y, z;};
struct Triangle{int v0, v1, v2;};
// -> above structures must contain 32-bit integers (-> Embree documentation).
//    Theoretically, these integers should be unsigned but the binary
//    representation until 2'147'483'647 is identical between signed/unsigned
//    integer.

// Initialise scene
RTCScene initializeScene(RTCDevice device, int* vertex_of_cell, int num_cell,
    vector<geom_point>& vertices){

    RTCScene scene = rtcNewScene(device);
    rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);
    RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Vertices (-> convert to float)
    Vertex* vertices_embree = (Vertex*) rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex),
        vertices.size());
    for (size_t i = 0; i < vertices.size(); i++) {
        vertices_embree[i].x = (float)vertices[i].x;
        vertices_embree[i].y = (float)vertices[i].y;
        vertices_embree[i].z = (float)vertices[i].z;
    }

    // Cell (triangle) indices to vertices
    Triangle* triangles_embree = (Triangle*) rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle),
        num_cell);
    for (int i = 0; i < num_cell; i++) {
        triangles_embree[i].v0 = vertex_of_cell[(0 * num_cell) + i];
        triangles_embree[i].v1 = vertex_of_cell[(1 * num_cell) + i];
        triangles_embree[i].v2 = vertex_of_cell[(2 * num_cell) + i];
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Commit geometry
    rtcCommitGeometry(geom);

    rtcAttachGeometry(scene, geom);
    rtcReleaseGeometry(geom);

    // Commit scene
    rtcCommitScene(scene);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time = end - start;
    std::cout << "Building BVH: " << time.count() << " s" << endl;

    return scene;

}

//#############################################################################
// Ray casting
//#############################################################################

bool castRay_occluded1(RTCScene scene, float ox, float oy, float oz, float dx,
    float dy, float dz, float dist_search){

    // Intersect context
    struct RTCIntersectContext context;
    // rtcInitIntersectContext() initializes the intersection context
    // to default values and should be called
    // to initialize every ray intersection context
    rtcInitIntersectContext(&context);

    // RTCRay - single ray structure - defines the ray layout for a single ray
    struct RTCRay ray;
    // origin members
    ray.org_x = ox;
    ray.org_y = oy;
    ray.org_z = oz;
    // direction vector
    ray.dir_x = dx;
    ray.dir_y = dy;
    ray.dir_z = dz;
    // ray segment
    ray.tnear = 0.0;
    //ray.tfar = std::numeric_limits<float>::infinity();
    ray.tfar = dist_search;

    // Intersect ray with scene - function that checks
    // wheter there is a hit with the scene
    rtcOccluded1(scene, &context, &ray);

    return (ray.tfar < 0.0);
}

//#############################################################################
// Horizon detection algorithms (horizon elevation angle)
//#############################################################################

//-----------------------------------------------------------------------------
// Guess horizon from previous azimuth direction
//-----------------------------------------------------------------------------

void ray_guess_const(float ray_org_x, float ray_org_y, float ray_org_z,
    double hori_acc, float dist_search,
    RTCScene scene, size_t &num_rays,
    double* horizon_cell, int horizon_cell_len, int azim_num,
    geom_vector sphere_normal, geom_vector north_direction,
    double azim_sin, double azim_cos,
    double elev_sin_1ha, double elev_cos_1ha,
    double elev_sin_2ha, double elev_cos_2ha){

    // ------------------------------------------------------------------------
    // First azimuth direction -> binary search
    // ------------------------------------------------------------------------

    // Lower/upper limit of 'binary search sector'
    double lim_low = -(M_PI / 2.0);  // [rad] (-90.0 deg)
    double lim_up = +(M_PI / 2.0);  // [rad] (+90.0 deg)

    // Initial elevation angle
    double elev_ang = (lim_low + lim_up) / 2.0;
    double ang_rot = 0.0;

    // Rotation axis (-> unit vector because cross product of two 
    // perpendicular unit vectors)
    geom_vector rot_axis = cross_product(north_direction, sphere_normal);

    // Initial ray direction
    geom_vector ray_dir;
    ray_dir.x = north_direction.x;
    ray_dir.y = north_direction.y;
    ray_dir.z = north_direction.z;

    // Rotate initial ray counterclockwise so that first azimuth sector
    // is centred around 0.0 deg (-> pointing towards North)
    double ang_shift = deg2rad(360.0 / (2.0 * azim_num));
    ray_dir = vector_rotation(ray_dir, sphere_normal, sin(ang_shift),
        cos(ang_shift));

    // Binary search
    bool hit;
    while ((lim_up - lim_low) > (2.0 * hori_acc)){

        hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
            ray_org_z, (float)ray_dir.x, (float)ray_dir.y, (float)ray_dir.z,
            dist_search);
        num_rays += 1;

        // Determine new elevation angle for ray sampling
        if (hit) {
            lim_low = elev_ang;
        } else {
            lim_up = elev_ang;
        }
        ang_rot = ((lim_low + lim_up) / 2.0) - elev_ang;
        elev_ang = (lim_low + lim_up) / 2.0;

        // Change elevation angle of ray direction
        ray_dir = vector_rotation(ray_dir, rot_axis, sin(ang_rot),
            cos(ang_rot));

  	}

    horizon_cell[0] = elev_ang;

    // ------------------------------------------------------------------------
    // Remaining azimuth directions -> guess horizon from previous azimuth
    // direction
    // ------------------------------------------------------------------------

    // Lower ray direction vector by 'hori_acc'
    ray_dir = vector_rotation(ray_dir, rot_axis, -elev_sin_1ha,
        elev_cos_1ha);  // sin(-x) == -sin(x), cos(x) == cos(-x)
    elev_ang -=hori_acc;

    for (int i = 1; i < horizon_cell_len; i++){

        // Azimuthal rotation of ray direction (clockwise; first to east)
        ray_dir = vector_rotation(ray_dir, sphere_normal, -azim_sin,
            azim_cos);  // sin(-x) == -sin(x), cos(x) == cos(-x)

        // Rotation axis (-> not a unit vector because vectors are not
        // necessarily perpendicular)
        rot_axis = cross_product(ray_dir, sphere_normal);
        unit_vector(rot_axis);

        // Find horizon with discrete ray sampling
        hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
            ray_org_z, (float)ray_dir.x, (float)ray_dir.y, (float)ray_dir.z,
            dist_search);
        num_rays += 1;
        if (hit) { // terrain hit -> increase elevation angle
            while (hit) {
                ray_dir = vector_rotation(ray_dir, rot_axis, elev_sin_2ha,
                    elev_cos_2ha);
                elev_ang += (2.0 * hori_acc);
                hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
                ray_org_z, (float)ray_dir.x, (float)ray_dir.y,
                (float)ray_dir.z, dist_search);
                num_rays += 1;
            }
            horizon_cell[i] = elev_ang - hori_acc;
        } else { // terrain not hit -> decrease elevation angle
            while (!hit) {
                ray_dir = vector_rotation(ray_dir, rot_axis, -elev_sin_2ha,
                    elev_cos_2ha);  // sin(-x) == -sin(x), cos(x) == cos(-x)
                elev_ang -= (2.0 * hori_acc);
                hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
                ray_org_z, (float)ray_dir.x, (float)ray_dir.y,
                (float)ray_dir.z, dist_search);
                num_rays += 1;
            }
            horizon_cell[i] = elev_ang + hori_acc;
        }
    }

}

//#############################################################################
// Sky View Factor algorithms
//#############################################################################

// Function pointer
double (*function_pointer)(double* horizon_cell, int horizon_cell_len,
    geom_vector normal);

//-----------------------------------------------------------------------------
// Visible sky fraction for horizontal plane
// (-> 'pure geometric sky view factor' in EXTPAR)
//-----------------------------------------------------------------------------

double pure_geometric_svf(double* horizon_cell, int horizon_cell_len,
    geom_vector normal){

	double svf = 0.0;
	for(int i = 0; i < horizon_cell_len; i++){
		svf += (1.0 - sin(horizon_cell[i]));
	}
	svf /= (double)horizon_cell_len;
    return svf;
}

//-----------------------------------------------------------------------------
// Sky view factor for horizontal plane
// (-> 'geometric scaled with sin(horizon)' in EXTPAR)
//-----------------------------------------------------------------------------

double geometric_svf_scaled_1(double* horizon_cell, int horizon_cell_len,
    geom_vector normal){

	double svf = 0.0;
	for(int i = 0; i < horizon_cell_len; i++){
		svf += (1.0 - (sin(horizon_cell[i]) * sin(horizon_cell[i])));
	}
	svf /= (double)horizon_cell_len;
	return svf;
}

//-----------------------------------------------------------------------------
// Sky view factor for horizontal plane additionally scaled with sin(horizon)
// (-> 'geometric scaled with sin(horizon)**2' in EXTPAR)
//-----------------------------------------------------------------------------

double geometric_svf_scaled_2(double* horizon_cell, int horizon_cell_len,
    geom_vector normal){

	double svf = 0.0;
	for(int i = 0; i < horizon_cell_len; i++){
		svf += (1.0 - (sin(horizon_cell[i]) * sin(horizon_cell[i])
		    * sin(horizon_cell[i])));
	}
	svf /= (double)horizon_cell_len;
	return svf;
}

//-----------------------------------------------------------------------------
// Sky view factor for sloped plane
//-----------------------------------------------------------------------------

double sky_view_factor(double* horizon_cell, int horizon_cell_len,
    geom_vector normal){

    /*Compute sky view factor considering the sloped terrain and Lambert's
    cosine law.

    Parameters
    ----------
    horizon_cell : array of double
        Array with horizon values [rad]
    horizon_cell_len : int
        Number of azimuth directions with horizon values
    normal : geom_vector
        Terrain surface normal (x, y, z) in local ENU coordinates [m]

    Returns
    -------
    svf : sky view factor [-]

	Reference
	---------
	Steger et al. (2022): HORAYZON v1.2: an efficient and flexible ray-tracing
	algorithm to compute horizon and sky view factor,
	https://doi.org/10.5194/gmd-15-6817-2022, Equation 11*/

	double svf = 0.0;
	double azim_spac = deg2rad(360.0) / (double)horizon_cell_len;
	double hori_plane, hori_elev;
	double azim_sin;
	double azim_cos;

	for(int i = 0; i < horizon_cell_len; i++){

		azim_sin = sin(i * (azim_spac));
		azim_cos = cos(i * (azim_spac));

		// Compute the plane-sphere intersection and select the maximum
		// between it and the elevation angle
		hori_plane = atan(-(normal.x / normal.z) * azim_sin
		    - (normal.y / normal.z) * azim_cos);
		if (horizon_cell[i] >= hori_plane){
			hori_elev = horizon_cell[i];
		} else {
			hori_elev = hori_plane;
		}

		svf += (normal.x * azim_sin + normal.y * azim_cos)
		    * ((M_PI / 2.0) - hori_elev - (sin(2.0 * hori_elev) / 2.0))
		    + normal.z * cos(hori_elev) * cos(hori_elev);
	}
	svf = (azim_spac / (2 * M_PI)) * svf;

	return svf;
}

//#############################################################################
// Main function
//#############################################################################

void horizon_svf_comp(double* vlon, double* vlat, float* topography_v,
    int num_vertex,
    double* clon, double* clat,  int* vertex_of_cell,
    int num_cell,
    float* horizon, float* skyview, int azim_num, int refine_factor,
    int svf_type){

    // Settings and constants
    double hori_acc = deg2rad(0.25);  // horizon accuracy [deg] (1.0)
    double ray_org_elev = 0.1;  // fix elevation offset [m] (0.1, 0.2, 0.5)
    float dist_search = 40000;  // horizon search distance [m] (50000.0)
    double rad_earth = 6371229.0;  // ICON/COSMO earth radius [m]

    // ------------------------------------------------------------------------
    // Pre-processing of data (coordinate transformation, etc.)
    // ------------------------------------------------------------------------

    std::cout << "---------------------------------------------" << endl;
    std::cout << "Horizon and SVF computation with Intel Embree" << endl;
    std::cout << "---------------------------------------------" << endl;

    // Adjust vertex indices (Fortran -> C; start with index 0)
    for (int i = 0; i < (num_cell * 3); i++){
        vertex_of_cell[i] -= 1;
    }

    std::cout << "Convert spherical to ECEF coordinates" << endl;

    // Circumcenters on surface of sphere (elevation = 0.0 m) (ECEF)
    vector<float> h_c(num_cell, 0.0);
    vector<geom_point> circumcenters = lonlat2ecef(clon, clat, &h_c[0],
        num_cell, rad_earth);

    // Cell vertices (ECEF)
    vector<geom_point> vertices = lonlat2ecef(vlon, vlat, topography_v,
        num_vertex, rad_earth);

    // Sphere normals for circumcenters (ECEF)
    vector<geom_vector> sphere_normals(num_cell);
    for (int i = 0; i < num_cell; i++){
        sphere_normals[i].x = circumcenters[i].x / rad_earth;
        sphere_normals[i].y = circumcenters[i].y / rad_earth;
        sphere_normals[i].z = circumcenters[i].z / rad_earth;
    }

    // North vectors for circumcenters (ECEF)
    vector<geom_vector> north_directions = north_direction(circumcenters,
        sphere_normals, rad_earth);

    // Origin of ENU coordinate system
    double lon_orig = 0.0;
    double lat_orig = 0.0;
    for (int i = 0; i < num_cell; i++){
        lon_orig += clon[i];
        lat_orig += clat[i];
    }
    lon_orig /= num_cell;
    lat_orig /= num_cell;

    // In-place transformation from ECEF to ENU
    std::cout << "Convert ECEF to ENU coordinates" << endl;
    std::cout << "Origin of ENU coordinate system: " << rad2deg(lat_orig)
        << " deg lat, " << rad2deg(lon_orig) << " deg lon" << endl;
    ecef2enu_point(circumcenters, lon_orig, lat_orig, rad_earth);
    ecef2enu_point(vertices, lon_orig, lat_orig, rad_earth);
    ecef2enu_vector(sphere_normals, lon_orig, lat_orig);
    ecef2enu_vector(north_directions, lon_orig, lat_orig);

    // ------------------------------------------------------------------------
    // Building of BVH
    // ------------------------------------------------------------------------

    RTCDevice device = initializeDevice();
    RTCScene scene = initializeScene(device, vertex_of_cell, num_cell,
        vertices);

    // ------------------------------------------------------------------------
    // Terrain horizon and sky view factor computation
    // ------------------------------------------------------------------------

    // Evaluated trigonometric functions for rotation along azimuth/elevation
    // angle
    int horizon_cell_len = azim_num * refine_factor;
    double azim_sin = sin(deg2rad(360.0) / (double)horizon_cell_len);
    double azim_cos = cos(deg2rad(360.0) / (double)horizon_cell_len);
    double elev_sin_1ha = sin(hori_acc);
    double elev_cos_1ha = cos(hori_acc);
    double elev_sin_2ha = sin(2.0 * hori_acc);
    double elev_cos_2ha = cos(2.0 * hori_acc);
    // Note: sin(-x) == -sin(x), cos(x) == cos(-x)

    // Select algorithm for sky view factor computation
    std::cout << "Sky View Factor computation algorithm: ";
    if (svf_type == 0) {
        std::cout << "pure geometric svf" << endl;
        function_pointer = pure_geometric_svf;
    } else if (svf_type == 1) {
        std::cout << "geometric scaled with sin(horizon)" << endl;
        function_pointer = geometric_svf_scaled_1;
    } else if (svf_type == 2) {
        std::cout << "geometric scaled with sin(horizon)**2" << endl;
        function_pointer = geometric_svf_scaled_2;
    } else if (svf_type == 3){
        std::cout << "SVF for sloped surface according to HORAYZON" << endl;
        function_pointer = sky_view_factor;
    }

    // ------------------------------------------------------------------------
    // Perform ray tracing
    // ------------------------------------------------------------------------
    auto start_ray = std::chrono::high_resolution_clock::now();
    size_t num_rays = 0;

    num_rays += tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, num_cell), 0.0,
    [&](tbb::blocked_range<size_t> r, size_t num_rays) {  // parallel

    //for(size_t i = 0; i < (size_t)num_cell; i++){ // serial
    for (size_t i=r.begin(); i<r.end(); ++i) {  // parallel

        // Indices of the triangle's vertices
        int idx1 = vertex_of_cell[i];
        int idx2 = vertex_of_cell[i + num_cell];
        int idx3 = vertex_of_cell[i + 2 * num_cell];

        // Two triangle edges
        geom_vector edge_1 = {vertices[idx2].x - vertices[idx1].x,
                              vertices[idx2].y - vertices[idx1].y,
                              vertices[idx2].z - vertices[idx1].z};
        geom_vector edge_2 = {vertices[idx3].x - vertices[idx1].x,
                              vertices[idx3].y - vertices[idx1].y,
                              vertices[idx3].z - vertices[idx1].z};

        // Triangle normal ('outward' with respect to the centre of the Earth)
        geom_vector triangle_normal = cross_product(edge_1, edge_2);
        unit_vector(triangle_normal);

        // Plane equation (a * x + b * y + c * z + d = 0)
        double d = - triangle_normal.x * vertices[idx1].x
                   - triangle_normal.y * vertices[idx1].y
                   - triangle_normal.z * vertices[idx1].z;

        // Intersection of plane and line
        // a * (c_x + t * n_x) + b * (c_y + t * n_y) + c * (c_z + t * n_z) + d
        // = 0
        double t = - (triangle_normal.x * circumcenters[i].x
                    + triangle_normal.y * circumcenters[i].y
                    + triangle_normal.z * circumcenters[i].z
                    + d)
                    / (triangle_normal.x * sphere_normals[i].x
                    + triangle_normal.y * sphere_normals[i].y
                    + triangle_normal.z * sphere_normals[i].z);

        // Circumcenters at tirangles' elevation
        circumcenters[i].x += t * sphere_normals[i].x;
        circumcenters[i].y += t * sphere_normals[i].y;
        circumcenters[i].z += t * sphere_normals[i].z;

        // Elevate origin for ray tracing by 'safety margin'
        float ray_org_x = (float)(circumcenters[i].x
            + sphere_normals[i].x * ray_org_elev);
        float ray_org_y = (float)(circumcenters[i].y
            + sphere_normals[i].y * ray_org_elev);
        float ray_org_z = (float)(circumcenters[i].z
            + sphere_normals[i].z * ray_org_elev);

        double* horizon_cell = new double[horizon_cell_len];  // [rad]

        // Compute terrain horizon
        ray_guess_const(ray_org_x, ray_org_y, ray_org_z,
            hori_acc, dist_search,
            scene, num_rays,
            horizon_cell, horizon_cell_len, azim_num,
            sphere_normals[i], north_directions[i],
            azim_sin, azim_cos,
            elev_sin_1ha, elev_cos_1ha,
            elev_sin_2ha, elev_cos_2ha);

        // Clip lower limit of terrain horizon values to 0.0
        for(int j = 0; j < horizon_cell_len; j++){
            if (horizon_cell[j] < 0.0){
                horizon_cell[j] = 0.0;
            }
        }

        // Compute mean horizon for sector and save in 'horizon' buffer
        for(int j = 0; j < azim_num; j++){
            double horizon_mean = 0.0;
            for(int k = 0; k < refine_factor; k++){
                horizon_mean += horizon_cell[(j * refine_factor) + k];
            }
            horizon[(j * num_cell) + i] = (float)(rad2deg(horizon_mean)
                / (double)refine_factor);
        }

        // Transform triangle surface normal from global to local ENU
        // coordinates
        geom_vector east_direction = cross_product(north_directions[i],
            sphere_normals[i]);
        double rotation_matrix[3][3] = {
            {east_direction.x, east_direction.y, east_direction.z},
            {north_directions[i].x, north_directions[i].y,
            north_directions[i].z},
            {sphere_normals[i].x, sphere_normals[i].y, sphere_normals[i].z}
            };
        geom_vector triangle_normal_local = vector_matrix_multiplication(
            triangle_normal, rotation_matrix);

        // Compute sky view factor and save in 'skyview' buffer
        skyview[i] = (float)function_pointer(horizon_cell, horizon_cell_len,
            triangle_normal_local);

        delete[] horizon_cell;

    }

    return num_rays;  // parallel
    }, std::plus<size_t>());  // parallel

    auto end_ray = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_ray = end_ray - start_ray;
    std::cout << "Ray tracing: " << time_ray.count() << " s" << endl;

    // Print number of rays needed for location and azimuth direction
    cout << "Number of rays shot: " << num_rays << endl;
    double ratio = (double)num_rays / (double)(num_cell * azim_num);
    printf("Average number of rays per cell and azimuth sector: %.2f \n",
        ratio);

}