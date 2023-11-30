#define _USE_MATH_DEFINES
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <exception>
#include <math.h>
#include <cmath>
#include <limits>
#include <stdio.h>
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
// Functions for Coordinate transformations
// Geodetic --> ECEF
// ECEF --> ENU
//#############################################################################

struct vertex_double{
	double x, y, z;
};

// Convert degree to radian
inline float deg2rad(float ang) {
	return ((ang / 180.0) * M_PI);
}

// Convert radian to degree
inline float rad2deg(float ang) {
	return ((ang / M_PI) * 180.0);
}

// Cross product
inline void cross_prod(double a_x, double a_y, double a_z, double b_x, double b_y,
	double b_z, double c_x, double c_y, double c_z) {
	c_x = a_y * b_z - a_z * b_y;
    c_y = a_z * b_x - a_x * b_z;
    c_z = a_x * b_y - a_y * b_x;
}

std::vector<vertex_double> lonlat2ecef(double* vlon, double* vlat, float* h, int len){

    /*Coordinate transformation from lon/lat to ECEF.

    Transformation of geodetic longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates.

    Parameters
    ----------
    vlon : array of double
        Array (one-dimensional) with geographic longitude [radian]
    vlat : array of double
        Array (one-dimensional) with geographic latitude [radian]
    h : array of float
        Array (one-dimensional) with elevation above sphere [metre]

    Returns
    -------
    xyz_ecef : vector of vertices
		vector of vertices with ECEF x, y, z coordinates [metre]
        
    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion*/

    // initialization
	vector<vertex_double> xyz_ecef(len);

    // Spherical coordinates
    double r = 6371229.0;  // earth radius (according to ICON documentation) [m]

	/*tbb::parallel_for(tbb::blocked_range<size_t>(0, len), //parallel
		[&](tbb::blocked_range<size_t> r){

	for (size_t i = r.begin(); i < r.end(); ++i){
		xyz_ecef[i].x = (r + h[i]) * cos(vlat[i]) * cos(vlon[i]);
		xyz_ecef[i].y = (r + h[i]) * cos(vlat[i]) * sin(vlon[i]);
		xyz_ecef[i].z = (r + h[i]) * sin(vlat[i]);
	}

	});*/

	for (int i = 0; i < len; i++){
		xyz_ecef[i].x = (r + h[i]) * cos(vlat[i]) * cos(vlon[i]);
		xyz_ecef[i].y = (r + h[i]) * cos(vlat[i]) * sin(vlon[i]);
		xyz_ecef[i].z = (r + h[i]) * sin(vlat[i]);
	}

    return xyz_ecef;
}

// -----------------------------------------------------------------------------

std::vector<vertex_double> ecef2enu(vector<double>& x_ecef, vector<double>& y_ecef, 
	vector<double>& z_ecef, int len, double lon_or, double lat_or){

    /*Coordinate transformation from ECEF to ENU.

    Transformation of earth-centered, earth-fixed (ECEF) to local tangent
    plane (ENU) coordinates.

    Parameters
    ----------
    x_ecef : array of double
        Array (one-dimensional) with ECEF x-coordinates [metre]
    y_ecef : array of double
        Array (one-dimensional) with ECEF y-coordinates [metre]
    z_ecef : array of double
        Array (one-dimensional) with ECEF z-coordinates [metre]

    Returns
    -------
	xyz_enu : vector of vertices
		vector of vertices with ENU x, y, z coordinates [metre] */

    // Check arguments
    double sin_lon, cos_lon, sin_lat, cos_lat;
	double x_ecef_or, y_ecef_or, z_ecef_or;
    vector<vertex_double> xyz_enu(len);

    // Check and change values to the latitude and longitude coordinates of the origin
    try{    
        if (lon_or < -180.0 || lon_or > 180.0) {
            throw invalid_argument("Value for 'lon_or' is outside of valid range");
        }
        if (lat_or < -90.0 || lat_or > 90.0) {
            throw invalid_argument("Value for 'lat_or' is outside of valid range");
        }
    }catch (const invalid_argument& e) {
        cerr << "Error: " << e.what() << endl;
    }

    const double r = 6371229.0; // earth radius (according to ICON documentation) [m]
    // origin coordinates
    x_ecef_or = r * cos(lat_or) * cos(lon_or);
    y_ecef_or = r * cos(lat_or) * sin(lon_or);
    z_ecef_or = r * sin(lat_or);

    // Trigonometric functions
    sin_lon = sin(lon_or);
    cos_lon = cos(lon_or);
    sin_lat = sin(lat_or);
    cos_lat = cos(lat_or);

    // Coordinate transformation

    // before it was 
    // for i in prange(len_0, nogil=True, schedule="static")
    // in which prange is to run in parallel in cython (don't know how to change it in c++)
    for (int i = 0; i < len; i++ ){
        xyz_enu[i].x = (- sin_lon * (x_ecef[i] - x_ecef_or) 
                    + cos_lon * (y_ecef[i] - y_ecef_or));
        xyz_enu[i].y = (- sin_lat * cos_lon * (x_ecef[i] - x_ecef_or)
                    - sin_lat * sin_lon * (y_ecef[i] - y_ecef_or)
                    + cos_lat * (z_ecef[i] - z_ecef_or));
        xyz_enu[i].z = (+ cos_lat * cos_lon * (x_ecef[i] - x_ecef_or)
                    + cos_lat * sin_lon * (y_ecef[i] - y_ecef_or)
                    + sin_lat * (z_ecef[i] - z_ecef_or));
    }

    return xyz_enu;
}



//#############################################################################
// Compute normal unit vector in ECEF coordinates
//#############################################################################

std::vector<vertex_double> surf_norm(double* lon, double* lat, int len){
    /*Compute surface normal unit vectors.

    Computation of surface normal unit vectors in earth-centered, earth-fixed
    (ECEF) coordinates.

    Parameters
    ----------
    lon : array of double
        Array (one-dimensional) with geographic longitude [degree]
    lat : array of double
        Array (one-dimensional) with geographic latitudes [degree]

    Returns
    -------
	vec_norm_ecef : vector of directions
		vector of directions of the surface normals in ECEF coordinates [metre]  
    
    Sources
    -------
    - https://en.wikipedia.org/wiki/N-vector */

    double sin_lon, cos_lon, sin_lat, cos_lat;
    vector<vertex_double> vec_norm(len);

    // Compute surface normals
    // for i in prange(len_0, nogil=True, schedule="static"):
    for (int i = 0; i < len; i++ ){
        sin_lon = sin(lon[i]);
        cos_lon = cos(lon[i]);
        sin_lat = sin(lat[i]);
        cos_lat = cos(lat[i]);
        vec_norm[i].x = cos_lat * cos_lon;
        vec_norm[i].y = cos_lat * sin_lon;
        vec_norm[i].z = sin_lat;
    }

    return vec_norm;
}

std::vector<vertex_double> north_dir(vector<vertex_double> points, int len, vector<vertex_double> vec_norm){
    /*Compute unit vectors pointing towards North.

    Computation unit vectors pointing towards North in earth-centered,
    earth-fixed (ECEF) coordinates. These vectors are perpendicular to surface
    normal unit vectors.

    Parameters
    ----------
    x_ecef : array of double
        Array (one-dimensionals) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (one-dimensional) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (one-dimensional) with ECEF z-coordinates [metre]

    vec_norm_ecef : vector of directions  
		vector (one-dimensional) of directions of surface normals 
		in ECEF coordinates [metre]

    Returns
    -------
	vec_north_ecef : vector of directions
				vector (one-dimensional) of directions towards north 
				in ECEF coordinates [metre]*/

    double np_x, np_y, np_z;
    double dir_north_x, dir_north_y, dir_north_z;
    double dot_pr, vec_proj_x, vec_proj_y, vec_proj_z, norm;
    vector<vertex_double> vec_north(len);

	double r = 6371229.0;  // earth radius [m]

    // Coordinates of North pole
    np_x = 0.0;
    np_y = 0.0;
    np_z = r;

    // Coordinate transformation
    for (int i = 0; i < len; i++){

        // Vector to North Pole
        dir_north_x = (np_x - points[i].x);
        dir_north_y = (np_y - points[i].y);
        dir_north_z = (np_z - points[i].z);

		if (i == 0){
			std::cout << "Vector to North Pole: (" << dir_north_x << ", " << dir_north_y << ", " 
				<< dir_north_z << ");" << endl;
		}

        // Project vector to North Pole on surface normal plane
        dot_pr = ((dir_north_x * vec_norm[i].x)
                  + (dir_north_y * vec_norm[i].y)
                  + (dir_north_z * vec_norm[i].z));
		
		if (i == 0){
			std::cout << "  Scalar product: " << dot_pr << ";" << endl;
		}

        vec_proj_x = dir_north_x - dot_pr * vec_norm[i].x;
        vec_proj_y = dir_north_y - dot_pr * vec_norm[i].y;
        vec_proj_z = dir_north_z - dot_pr * vec_norm[i].z;

		if (i == 0){
			std::cout << "Vector projection on surface normal plane: (" 
				<< vec_proj_x << ", " << vec_proj_y << ", " << vec_proj_z << ");" << endl;
		}

        // Normalise vector
        norm = sqrt(vec_proj_x * vec_proj_x + vec_proj_y * vec_proj_y + vec_proj_z * vec_proj_z);

		if (i == 0){
			std::cout << "Norm: " << norm << endl;
		}

		vec_north[i].x = (vec_proj_x / norm);
        vec_north[i].y = (vec_proj_y / norm);
        vec_north[i].z = (vec_proj_z / norm);
		if (i == 0){
			std::cout << "Vector projection on surface normal plane: (" 
				<< vec_north[i].x << ", " << vec_north[i].x << ", " << vec_north[i].x << ");" << endl;
		}		

	}	

    return vec_north;
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

// Structure for triangles
struct Triangle{ int v0, v1, v2; };
// -> above structures must contain 32-bit integers (-> Embree documentation).
//    Theoretically, these integers should be unsigned but the binary
//    representation until 2'147'483'647 is identical between signed/unsigned
//    integer.


// Initialise scene
RTCScene initializeScene(RTCDevice device, float* vert_grid, 
	int* vertex_of_cell, int num_vert, int num_tri){ 

	RTCScene scene = rtcNewScene(device);
  	rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);

	//std::cout << *(vert_grid + 3) << " " << *(vertex_of_cell + 3) << " " << num_vert << " " << num_tri << endl << endl;

  	RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);  	
  	rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
  		RTC_FORMAT_FLOAT3, vert_grid, 0, 3*sizeof(float), num_vert);  	
	
  	/*rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0,
  		RTC_FORMAT_UINT3, vertex_of_cell, 0, 3*sizeof(int32_t), num_tri);*/ 

	// assign a INDEX data buffer to the geometry
    Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle),
        num_tri);

	for (int i = 0; i < num_tri; i++) {
			triangles[i].v0 = vertex_of_cell[3 * i] - 1;
			triangles[i].v1 = vertex_of_cell[3 * i + 1] - 1;
			triangles[i].v2 = vertex_of_cell[3 * i + 2] - 1;

			//std::cout << "triangle nr. " << i << " : (" << triangles[i].v0 << ", " << triangles[i].v1 << ", " << triangles[i].v2 << ");  ";
	}
	//std::cout << endl;

	auto start = std::chrono::high_resolution_clock::now();

	// Commit geometry
	rtcCommitGeometry(geom);

	rtcAttachGeometry(scene, geom);
	rtcReleaseGeometry(geom);

	// removed the thing for outer simplified domain
	// Commit scene
	rtcCommitScene(scene);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << "BVH build time: " << time.count() << " s" << endl;

	return scene;

}

vector<float> pad_buffer(vector<float> buffer){
    /*Padding of geometry buffer.

    Pads geometric buffer to make it conformal with 16-byte SSE load
    instructions.

    Parameters
    ----------
    buffer : ndarray
        Array (one-dimensional) with geometry buffer [arbitrary]

    Returns
    -------
    buffer : ndarray
        Array (one-dimensional) with padded geometry buffer [arbitrary]

    Notes
    -----
    This function ensures that vertex buffer size is divisible by 16 and hence
    conformal with 16-byte SSE load instructions (see Embree documentation;
    section 7.45 rtcSetSharedGeometryBuffer).*/

    int add_elem = 16;
    if ((sizeof(buffer) % 16) != 0){
        add_elem += ((16 - (sizeof(buffer) % 16)));
	}
	for(int i=0; i < add_elem; i++){
    	buffer.push_back((float)0.0);	
	}

	return buffer;
}
//#############################################################################
// Ray casting
//#############################################################################

//-----------------------------------------------------------------------------
// Cast single ray (occluded; higher performance)
//-----------------------------------------------------------------------------

bool castRay_occluded1(RTCScene scene, float ox, float oy, float oz, float dx,
	float dy, float dz, float dist_search) {
  
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
  	ray.tfar = std::numeric_limits<float>::infinity();
  	//ray.tfar = dist_search;

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
	size_t azim_num, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim,
	RTCScene scene, size_t &num_rays, vector<float> hori_buffer,
	float norm_x, float norm_y, float norm_z,
	float north_x, float north_y, float north_z,
	float azim_sin, float azim_cos, int cell) {

	// ------------------------------------------------------------------------
  	// First azimuth direction (binary search - faster)
  	// ------------------------------------------------------------------------

	float param;
  	float lim_up = elev_ang_up_lim;
  	float lim_low = elev_ang_low_lim;

	// elev_ang in my case is the angle that defines
	// the rotation angle for Rodrigues' formula
  	float elev_ang = 0; // (abs(lim_up) - abs(lim_low)) / 2.0;
	float elev_sin, elev_cos;

	// final_ang is the angle that describes 
	// the final horizon angle
	float final_ang = elev_ang;

	float counterclock_prod_x, counterclock_prod_y, counterclock_prod_z;

	// cross product to then have the normal axis 
	// that allows a counter-clockwise rotation
	counterclock_prod_x = north_y * norm_z - north_z * norm_y;
	counterclock_prod_y = north_z * norm_x - north_x * norm_z;
	counterclock_prod_z = north_x * norm_y - north_y * norm_x;

	float new_dir_x, new_dir_y, new_dir_z;
	new_dir_x = north_x;
	new_dir_y = north_y;
	new_dir_z = north_z;

	int hit_num = 0;
  	while (max(lim_up - final_ang,
  		final_ang - lim_low) > hori_acc) {

  		bool hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  			ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  			dist_search);
  		num_rays += 1;
  		
  		if (hit) {

			hit_num++;
  			lim_low = final_ang;
			elev_ang = (lim_up - lim_low) / 2.0;

  		} else {

			hit_num++;
  			lim_up = final_ang;		
			elev_ang = - (lim_up - lim_low) / 2.0;

  		}

		elev_sin = sin(elev_ang);
		elev_cos = cos(elev_ang);

		// Rodrigues' rotation formula
		param = (1 - elev_cos) * 
			(counterclock_prod_x * new_dir_x + counterclock_prod_y * new_dir_y + counterclock_prod_z * new_dir_z);

		new_dir_x = new_dir_x * elev_cos + 
			(counterclock_prod_y * new_dir_z - counterclock_prod_z * new_dir_y) * elev_sin + 
			counterclock_prod_x * param;
		new_dir_y = new_dir_y * elev_cos + 
			(counterclock_prod_z * new_dir_x - counterclock_prod_x * new_dir_z) * elev_sin + 
			counterclock_prod_y * param;
		new_dir_z = new_dir_z * elev_cos + 
			(counterclock_prod_x * new_dir_y - counterclock_prod_y * new_dir_x) * elev_sin + 
			counterclock_prod_z * param;
		
		final_ang += elev_ang;
		
	/*if ((hit_num==1) || (hit_num == 2) || (hit_num == 3) || (hit_num == 4) || (hit_num == 5) || (hit_num == 6)){
		std::cout << "lim_up = " << rad2deg(lim_up) << endl;
		std::cout << "lim_low = " << rad2deg(lim_low) << endl;
		std::cout << "elev_ang = " << rad2deg(elev_ang) << endl;
		std::cout << "final_ang = " << rad2deg(final_ang) << endl; 			
	}*/		
  	}


  	hori_buffer[0] = final_ang;
	std::cout << "Horizon first azimuth: " << rad2deg(hori_buffer[0]) << endl;
	// initialize the variable containing the previous azimuth value
  	float prev_elev_ang = final_ang;

	// ------------------------------------------------------------------------
	// Remaining azimuth directions (guess horizon from previous
	// azimuth direction)
	// ------------------------------------------------------------------------
	azim_sin = - azim_sin; // to have clockwise rotation

	for (size_t k = 1; k < azim_num; k++){
		
		// Rodrigues' rotation formula TO CHANGE THE AZIMUTH
		// the norm_ vector is the rotation axis (so we'll rotate counter clockwise);  
		// the vector that must be rotated is the previous direction
		float param = (1 - azim_cos) * 
			(norm_x * new_dir_x + norm_y * new_dir_y + norm_z * new_dir_z);

		new_dir_x = new_dir_x * azim_cos + 
			(norm_y * new_dir_z - norm_z * new_dir_y) * azim_sin + 
			norm_x * param;
		new_dir_y = new_dir_y * azim_cos + 
			(norm_z * new_dir_x - norm_x * new_dir_z) * azim_sin + 
			norm_y * param;
		new_dir_z = new_dir_z * azim_cos + 
			(norm_x * new_dir_y - norm_y * new_dir_x) * azim_sin + 
			norm_z * param;

		// now focus on elevation rotation
		// cross product to then have a counter-clockwise rotation
		counterclock_prod_x = new_dir_y * norm_z - new_dir_z * norm_y;
		counterclock_prod_y = new_dir_z * norm_x - new_dir_x * norm_z;
		counterclock_prod_z = new_dir_x * norm_y - new_dir_y * norm_x;

		// Move upwards to check if the horizon is higher
		float delta = 0.0175; // 1 degree in [radians]
		float sin_delta = sin(delta);
		float cos_delta = cos(delta);
		bool hit = true;
		int count = 0;

		// discrete ray sampling
		while (hit) {	
			final_ang += delta;

			// Rodrigues' rotation formula
			param = (1 - cos_delta) * 
				(counterclock_prod_x * new_dir_x + counterclock_prod_y * new_dir_y + counterclock_prod_z * new_dir_z);

			new_dir_x = new_dir_x * cos_delta + 
				(counterclock_prod_y * new_dir_z - counterclock_prod_z * new_dir_y) * sin_delta + 
				counterclock_prod_x * param;
			new_dir_y = new_dir_y * cos_delta + 
				(counterclock_prod_z * new_dir_x - counterclock_prod_x * new_dir_z) * sin_delta + 
				counterclock_prod_y * param;
			new_dir_z = new_dir_z * cos_delta + 
				(counterclock_prod_x * new_dir_y - counterclock_prod_y * new_dir_x) * sin_delta + 
				counterclock_prod_z * param;
			//std::cout << new_dir_x << " " << new_dir_y << " " << new_dir_z << endl;

  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);

			//std::cout << hit << endl;	
  			num_rays += 1;
			// increase the count for number of hitting 
			// starting from the previous elevation angle
  			count += 1;		
		}

		// Move downwards to check if the horizon is lower
		hit = false;

		// discrete ray sampling until it hits the topography
		while (!hit) {
			final_ang -= delta;			
			// Rodrigues' rotation formula
			param = (1 - cos_delta) * 
				((- counterclock_prod_x) * new_dir_x + (- counterclock_prod_y) * new_dir_y + (- counterclock_prod_z) * new_dir_z);

			new_dir_x = new_dir_x * cos_delta + 
				((- counterclock_prod_y) * new_dir_z - (- counterclock_prod_z) * new_dir_y) * sin_delta + 
				(- counterclock_prod_x) * param;
			new_dir_y = new_dir_y * cos_delta + 
				((- counterclock_prod_z) * new_dir_x - (- counterclock_prod_x) * new_dir_z) * sin_delta + 
				(- counterclock_prod_y) * param;
			new_dir_z = new_dir_z * cos_delta + 
				((- counterclock_prod_x) * new_dir_y - (- counterclock_prod_y) * new_dir_x) * sin_delta + 
				(- counterclock_prod_z) * param;

  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);
  			num_rays += 1;	
		}

  		hori_buffer[cell*azim_num + k] = final_ang;
		std::cout << "Horizon azimuth nr." << k + 1 << ": " << rad2deg(hori_buffer[cell*azim_num + k]) << endl;
	}

}

//#############################################################################
// MAIN FUNCTION
//#############################################################################


void horizon_svf_comp(double* vlon, double* vlat, float* topography_v, 
	int n_vert, 
	double* clon, double* clat,  int* vertex_of_cell, uint8_t* mask, 
    int cell, 
	float* horizon, float* skyview, int nhori){
    
    int nv = 3;  // number of vertices per cell

    std::cout << "--------------------------------------------------------" << endl;
	std::cout << "Horizon and Sky View Factor(SVF) computation with Intel Embree" << endl;
	std::cout << "--------------------------------------------------------" << endl;

	vector<vertex_double> vertices(n_vert), circumcenters(cell);
	//vector<vertex_float> vertices_f(n_vert), circumcenters_f(cell);
    int idx1, idx2, idx3;
    vector<float> h_c(cell, 0.0);

    double dir1_x, dir1_y, dir1_z, dir2_x, dir2_y, dir2_z;
    double d, param, lon_or, lat_or;;

	// circumcenters at elevation 0
    circumcenters = lonlat2ecef(clon, clat, &h_c[0], cell);

	// vertices at correct elevation
	vertices = lonlat2ecef(vlon, vlat, topography_v, n_vert);

	// create the vector with vertices and the indices
	// for the creation of the RTC buffer
	vector<float> vert_buffer;	
	vector<int> vertex_of_cell_buffer;

  	// Compute vectors normal to the surface in the circumcenters
	vector<vertex_double> vec_norm = circumcenters;

	double abs_val;
	for(int i = 0; i < cell; i++){
		//normalization of the surface normals
		abs_val = sqrt(circumcenters[i].x * circumcenters[i].x + circumcenters[i].y * circumcenters[i].y + circumcenters[i].z * circumcenters[i].z);
		vec_norm[i].x /= abs_val;
		vec_norm[i].y /= abs_val;
		vec_norm[i].z /= abs_val;

		//creation of the buffer of indices needed for the rtc scene
		vertex_of_cell_buffer.push_back(vertex_of_cell[i]);
		vertex_of_cell_buffer.push_back(vertex_of_cell[i + cell]);
		vertex_of_cell_buffer.push_back(vertex_of_cell[i + 2*cell]);
	} 

	// north vectors in the circumcenters
	vector<vertex_double> vec_north = north_dir(circumcenters, cell, vec_norm);

	for (int i = 0; i < n_vert; i++){
		//creation of the buffer of vertex needed for the rtc scene
		vert_buffer.push_back((float)vertices[i].x);
		vert_buffer.push_back((float)vertices[i].y);
		vert_buffer.push_back((float)vertices[i].z);

		lon_or += vlon[i];
		lat_or += vlat[i];
	}

	//final origin for ENU coords
	lon_or /= n_vert;
	lat_or /= n_vert;

	// pad for vert_buffer
	vert_buffer = pad_buffer(vert_buffer);
	// pad for vertex_of_cell_buffer
    int add_elem = 16;
    if ((sizeof(vertex_of_cell_buffer) % 16) != 0){
        add_elem += ((16 - (sizeof(vertex_of_cell_buffer) % 16)));
	}
	for(int i=0; i < add_elem; i++){
    	vertex_of_cell_buffer.push_back((int)0);	
	}


	// upper  and lower limits for elevation angle [radians]
  	float elev_ang_up_lim = deg2rad(90);  
	float elev_ang_low_lim = deg2rad(-90);

	float hori_acc_deg = 1; // horizon accuracy [degrees]
	float hori_acc = deg2rad(hori_acc_deg);
	float dist_search = 20000;  // [meter]

	// number of azimuth sectors
	float hori_buffer_size_max = 1.5;	

	// fix elevation offset [meter]
	float ray_org_elev= 1; //0.01;
	// fix angle for azimuth rotation [radians]
	size_t azim_num = nhori;
	float azim_sin = sin((deg2rad(360)/azim_num)); 
	float azim_cos = cos((deg2rad(360)/azim_num));

	// horizon buffer
	vector<float> hori_buffer(azim_num*cell, 0.0);

  	//std::cout << "Horizon detection algorithm: horizon guess from previous azimuth direction." << endl;

  	int num_gc = 0; // number of grid triangles visited 
  	for (size_t i = 0; i < (size_t)(cell); i++) {
  		if (mask[i] == 1) {
  			num_gc += 1;
  		}
  	}
  	printf("Number of grid cells for which horizon is computed: %d \n",
  		num_gc);
  	std::cout << "Fraction of total number of grid cells: " << ((float)num_gc 
  		/ (float)cell * 100.0) << " %" << endl;


	// compute the size of the buffer for the current grid
	float hori_buffer_size = (((float)cell 
		* (float)azim_num * 4.0) / pow(10.0, 9.0));
	std::cout << "Total memory required for horizon output: " 
			  << hori_buffer_size << " GB" << endl;

	size_t num_rays = 0;
  	std::chrono::duration<double> time_ray = std::chrono::seconds(0);
  	std::chrono::duration<double> time_out = std::chrono::seconds(0);

	// --------------------------------------------------------
	// Horizon computation with Intel Embree 
	// --------------------------------------------------------

  	// Initialization and time counting
  	auto start_ini = std::chrono::high_resolution_clock::now();

  	RTCDevice device = initializeDevice();
  	RTCScene scene = initializeScene(device, &vert_buffer[0], &vertex_of_cell_buffer[0], n_vert, cell);

  	auto end_ini = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = end_ini - start_ini;
  	std::cout << "Total initialisation time: " << time.count() << " s" << endl;

    // ------------------------------------------------------------------------
  	// Compute and save horizon in one iteration
    // ------------------------------------------------------------------------

    if (hori_buffer_size <= hori_buffer_size_max) {
    
    	cout << "Compute and save horizon in one iteration" << endl;

    	// --------------------------------------------------------------------
  		// Perform ray tracing
    	// --------------------------------------------------------------------

  		auto start_ray = std::chrono::high_resolution_clock::now();

		for(int i = 0; i < cell; i++){
			// define indices of the vertices of the triangle
			// keep in mind that vertex_of_cell values start from 1 not 0
			idx1 = vertex_of_cell[i] - 1;
			idx2 = vertex_of_cell[i + cell] - 1;
			idx3 = vertex_of_cell[i + 2 * cell] - 1;

			// compute two directions that will allow 
			// to generate the plane of each triangle
			dir1_x = vertices[idx2].x - vertices[idx1].x;
			dir1_y = vertices[idx2].y - vertices[idx1].y;
			dir1_z = vertices[idx2].z - vertices[idx1].z;
			dir2_x = vertices[idx3].x - vertices[idx1].x;
			dir2_y = vertices[idx3].y - vertices[idx1].y;
			dir2_z = vertices[idx3].z - vertices[idx1].z;

			// compute coeffiecients of the normal of the trg
			double norm_x = dir1_y * dir2_z - dir1_z * dir2_y;
			double norm_y = dir1_z * dir2_x - dir1_x * dir2_z;
			double norm_z = dir1_x * dir2_y - dir1_y * dir2_x;

			/*double abs_val = sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z);

			norm_x = norm_x/abs_val;
			norm_y = norm_y/abs_val;
			norm_z = norm_z/abs_val;

			if( i == 0 || i == 1 || i == 2){
				std::cout << "(" << norm_x << ", " << norm_y << ", " << norm_z << ")" << " ";			
			}*/
			
			// compute the constant term of the plane 
			d = - norm_x * vertices[idx1].x - norm_y * vertices[idx1].y - norm_z * vertices[idx1].z;
			
			// compute intersection between the line and the plane
			// note that in our case, since the origin is (0,0,0)
			// the line that we have to intersect is simply defined 
			// by the coordinates of the circumcenter 
			param = - d / (norm_x * circumcenters[i].x + norm_y * circumcenters[i].y + norm_z * circumcenters[i].z);

			// circumcenters at tirangles' elevation
			circumcenters[i].x *= param; 
			circumcenters[i].y *= param; 
			circumcenters[i].z *= param; 

			// Transfor the Norm and North vectors from double to float
			float vec_norm_x = (float)vec_norm[i].x;
			float vec_norm_y = (float)vec_norm[i].y;
			float vec_norm_z = (float)vec_norm[i].z;

			float vec_north_x = (float)vec_north[i].x;
			float vec_north_y = (float)vec_north[i].y;
			float vec_north_z = (float)vec_north[i].z;      

			float circ_x = (float)circumcenters[i].x;
			float circ_y = (float)circumcenters[i].y;
			float circ_z = (float)circumcenters[i].z;  

			if (mask[i] == 1){

  					// Ray origin ECEF CASE
					
  					float ray_org_x = circ_x 
  						+ vec_norm_x * ray_org_elev;
  					float ray_org_y = circ_y 
  						+ vec_norm_y * ray_org_elev;
  					float ray_org_z = circ_z 
  						+ vec_norm_z * ray_org_elev;

					//std::cout << "Ray origin: (" << ray_org_x << ", " << ray_org_y << ", " << ray_org_z << ")" << endl;
					
					//std::cout << "Vec norm ecef: (" << vec_norm_ecef_x[i] << ", " << vec_norm_ecef_y[i] << ", " << vec_norm_ecef_z[i] << ")" << endl;
					//std::cout << "Vec north ecef: (" << vec_north_ecef_x[i] << ", " << vec_north_ecef_y[i] << ", " << vec_north_ecef_z[i] << ")" << endl;
					
					ray_guess_const(ray_org_x, ray_org_y, ray_org_z, azim_num, 
						hori_acc, dist_search, elev_ang_low_lim, elev_ang_up_lim, 
						scene, num_rays, hori_buffer, vec_norm_x, 
						vec_norm_y, vec_norm_z, vec_north_x, 
						vec_north_y, vec_north_z, azim_sin, azim_cos, i);	

					std::cout << endl;

			}else{

			}	
		
		}

  		auto end_ray = std::chrono::high_resolution_clock::now();
  		time_ray += (end_ray - start_ray);
	}

    return ;
}