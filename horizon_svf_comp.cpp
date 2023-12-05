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
// Coordinate transformation, north vector
//#############################################################################

std::vector<geom_point> lonlat2ecef(double* vlon, double* vlat,
    float* elevation, int num_point, double rad_earth){

    /*Transformation of geographic longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates. Assume spherical Earth.

    Parameters
    ----------
    vlon : array of double
        Array (one-dimensional) with geographic longitude [radian]
    vlat : array of double
        Array (one-dimensional) with geographic latitude [radian]
    elevation : array of float
        Array (one-dimensional) with elevation above sphere [metre]

    Returns
    -------
    points_ecef : vector of type <geom_point>
		vector of points (x, y, z) [metre]*/

	vector<geom_point> points_ecef(num_point);
	for (int i = 0; i < num_point; i++){
		points_ecef[i].x = (rad_earth + elevation[i]) * cos(vlat[i])
		    * cos(vlon[i]);
		points_ecef[i].y = (rad_earth + elevation[i]) * cos(vlat[i])
		    * sin(vlon[i]);
		points_ecef[i].z = (rad_earth + elevation[i]) * sin(vlat[i]);
	}

    return points_ecef;
}

// ----------------------------------------------------------------------------

std::vector<geom_vector> north_direction(vector<geom_point> points,
    int num_vector, vector<geom_vector> vector_normal, double rad_earth){
    /*Compute unit vectors pointing towards North in earth-centered,
    earth-fixed (ECEF) coordinates. These vectors are perpendicular to surface
    normal unit vectors.*/
    
    // North pole in ECEF
    geom_vector v_p = {0.0, 0.0, rad_earth};

    // initialization
	vector<geom_vector> vector_north(num_vector);

    geom_vector v_n, v_j;
    double dot_prod, v_j_mag;
    for (int i = 0; i < num_vector; i++){
        v_n.x = v_p.x - points[i].x;
        v_n.y = v_p.y - points[i].y;
        v_n.z = v_p.z - points[i].z;
        dot_prod = ((v_n.x * vector_normal[i].x)
            + (v_n.y * vector_normal[i].y)
            + (v_n.z * vector_normal[i].z));
        v_j.x = v_n.x - dot_prod * vector_normal[i].x;
        v_j.y = v_n.y - dot_prod * vector_normal[i].y;
        v_j.z = v_n.z - dot_prod * vector_normal[i].z;
        v_j_mag = sqrt(v_j.x * v_j.x + v_j.y * v_j.y + v_j.z * v_j.z);
        vector_north[i].x = v_j.x / v_j_mag;
        vector_north[i].y = v_j.y / v_j_mag;
        vector_north[i].z = v_j.z / v_j_mag;
    }

    return vector_north;
}

// ----------------------------------------------------------------------------

void ecef2enu_point(vector<geom_point>& points_ecef, double lon_orig,
    double lat_orig, double rad_earth){

    double sin_lon = sin(lon_orig);
    double cos_lon = cos(lon_orig);
    double sin_lat = sin(lat_orig);
    double cos_lat = cos(lat_orig);

    double x_ecef_orig = rad_earth * cos(lat_orig) * cos(lon_orig);
    double y_ecef_orig = rad_earth * cos(lat_orig) * sin(lon_orig);
    double z_ecef_orig = rad_earth * sin(lat_orig);

    double x_enu, y_enu, z_enu;
    for (size_t i = 0; i < points_ecef.size(); i++){
        x_enu = - sin_lon * (points_ecef[i].x - x_ecef_orig)
            + cos_lon * (points_ecef[i].y - y_ecef_orig);
        y_enu = - sin_lat * cos_lon * (points_ecef[i].x - x_ecef_orig)
            - sin_lat * sin_lon * (points_ecef[i].y - y_ecef_orig)
            + cos_lat * (points_ecef[i].z - z_ecef_orig);
        z_enu = + cos_lat * cos_lon * (points_ecef[i].x - x_ecef_orig)
            + cos_lat * sin_lon * (points_ecef[i].y - y_ecef_orig)
            + sin_lat * (points_ecef[i].z - z_ecef_orig);
        points_ecef[i].x = x_enu;
        points_ecef[i].y = y_enu;
        points_ecef[i].z = z_enu;
    }

}

// ----------------------------------------------------------------------------

void ecef2enu_vector(vector<geom_vector>& vector_ecef, double lon_orig,
    double lat_orig){

    double sin_lon = sin(lon_orig);
    double cos_lon = cos(lon_orig);
    double sin_lat = sin(lat_orig);
    double cos_lat = cos(lat_orig);

    double x_enu, y_enu, z_enu;
    for (size_t i = 0; i < vector_ecef.size(); i++){
        x_enu = - sin_lon * vector_ecef[i].x
            + cos_lon * vector_ecef[i].y;
        y_enu = - sin_lat * cos_lon * vector_ecef[i].x
            - sin_lat * sin_lon * vector_ecef[i].y
            + cos_lat * vector_ecef[i].z;
        z_enu = + cos_lat * cos_lon * vector_ecef[i].x
            + cos_lat * sin_lon * vector_ecef[i].y
            + sin_lat * vector_ecef[i].z;
        vector_ecef[i].x = x_enu;
        vector_ecef[i].y = y_enu;
        vector_ecef[i].z = z_enu;
    }

}

//#############################################################################
// Miscellaneous
//#############################################################################

// Convert degree to radian
inline double deg2rad(double ang) {
	return ((ang / 180.0) * M_PI);
}

// Convert radian to degree
inline double rad2deg(double ang) {
	return ((ang / M_PI) * 180.0);
}

// Cross product
/* 
inline void cross_prod(double a_x, double a_y, double a_z,
    double b_x, double b_y, double b_z,
    double c_x, double c_y, double c_z) {
	c_x = a_y * b_z - a_z * b_y;
    c_y = a_z * b_x - a_x * b_z;
    c_z = a_x * b_y - a_y * b_x;
}
 */

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

// ----------------------------------------------------------------------------

vector<float> pad_buffer(vector<float> buffer){
    /*Pads geometric buffer to make it conformal with 16-byte SSE load
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

    std::cout << "sizeof(buffer): " << sizeof(buffer) << endl;
    std::cout << "buffer.size(): " << buffer.size() << endl;

    int add_elem = 16;
    if ((sizeof(buffer) % 16) != 0){
        add_elem += floor(((16 - (sizeof(buffer) % 16))) / 4);
	}
	
	std::cout << "sizeof(buffer): " << sizeof(buffer) << endl;
	std::cout << "add_elem: " << add_elem << endl;
	
	for(int i=0; i < add_elem; i++){
    	buffer.push_back((float)0.0);	
	}
	
	std::cout << "sizeof(buffer): " << sizeof(buffer) << endl;

	return buffer;
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
	RTCScene scene, size_t &num_rays, vector<float>& hori_buffer,
	float norm_x, float norm_y, float norm_z,
	float north_x, float north_y, float north_z,
	float azim_sin, float azim_cos, int num_cell) {

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


  	hori_buffer[num_cell * azim_num] = final_ang;
	std::cout << "Horizon first azimuth: " << rad2deg(hori_buffer[num_cell * azim_num]) << endl;
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
		bool hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);
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
			//std::cout << "(" << new_dir_x << " " << new_dir_y << " " << new_dir_z << ")" << endl;

  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);

			//std::cout << hit << endl;	
  			num_rays += 1;
			// increase the count for number of hitting 
			// starting from the previous elevation angle
  			count += 1;		
			//std::cout << "Final angle: " << rad2deg(final_ang) << endl;
		}

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
			//std::cout << "(" << new_dir_x << " " << new_dir_y << " " << new_dir_z << ")" << endl;

  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);

			//std::cout << hit << endl;
  			num_rays += 1;	
			//std::cout << "Final angle: " << rad2deg(final_ang) << endl;
		}

  		//hori_buffer[num_cell * azim_num + k] = final_ang;
		//std::cout << "Horizon azimuth nr." << k + 1 << ": " << rad2deg(hori_buffer[num_cell*azim_num + k]) << endl;
 		hori_buffer[num_cell * azim_num + k] = final_ang;
		std::cout << "Horizon azimuth nr." << k + 1 << ": " << rad2deg(hori_buffer[num_cell * azim_num + k]) << endl;
	}

}


float _sky_view_factor(int azim_num, vector<float> hori_buffer, int num_cell,
                        float vec_norm_x, float vec_norm_y, float vec_norm_z, int svf_type){

    // Sky view factor (SVF) computation.
	float svf = 0;

	if(svf_type == 1){

		// pure geometric skyview-factor
		for(int k = 0; k < azim_num; k++){
			svf += (1 - sin(hori_buffer[num_cell * azim_num + k]));	
			std::cout << (num_cell*azim_num +k) << " ";	
		}
		std::cout << endl;
		svf /= azim_num;

	}else if (svf_type == 2){

		// geometric scaled with sin(horizon)
		for(int k = 0; k < azim_num; k++){
			svf += (1 - (sin(hori_buffer[num_cell * azim_num + k]) 
					* sin(hori_buffer[num_cell * azim_num + k])));
		}
		svf /= azim_num;

	}else if (svf_type == 3){

		// geometric scaled with sin(horizon)
		for(int k = 0; k < azim_num; k++){
			svf += (1 - (sin(hori_buffer[num_cell * azim_num + k]) 
						* sin(hori_buffer[num_cell * azim_num + k]) 
						* sin(hori_buffer[num_cell * azim_num + k])));

		}
		svf /= azim_num;

	}else{
		
		// Steger et al. Sky View Factor Computation
		float azim_spac = deg2rad(360)/azim_num;
		float hori_plane, hori_elev;
		float azim_sin = 0;
		float azim_cos = 0;

		for(int k = 0; k < azim_num; k++){
			azim_sin = sin(k * (azim_spac));
			azim_cos = cos(k * (azim_spac));
			
			// compute plane-sphere intersection and
			// select the max between it and the elevation angle
			hori_plane = atan( - (vec_norm_x / vec_norm_z) * azim_sin
								- (vec_norm_y / vec_norm_z) * azim_cos );
			if (hori_buffer[num_cell * azim_num + k] >= hori_plane){
				hori_elev = hori_buffer[num_cell * azim_num + k];
			}else{
				hori_elev = hori_plane;
			}

			svf += (vec_norm_x * azim_sin + vec_norm_y * azim_cos) *
					((M_PI / 2) - hori_elev - (sin(2 * hori_elev) / 2 )) +
					vec_norm_z * cos(hori_elev) * cos(hori_elev);

		}
		svf = (azim_spac / (2 * M_PI)) * svf;

	}
    return svf;
}

//#############################################################################
// Main function
//#############################################################################

void horizon_svf_comp(double* vlon, double* vlat, float* topography_v,
	int num_vertex,
	double* clon, double* clat,  int* vertex_of_cell, uint8_t* mask,
    int num_cell,
	float* horizon, float* skyview, int nhori, int svf_type){

    // Settings and constants
    float hori_acc_deg = 1;  // horizon accuracy [degree]
    float ray_org_elev= 0.5;  // fix elevation offset [m]; before: 0.1
    float dist_search = 50000;  // horizon search distance [m]
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

    // Number of considered cells
  	int num_cell_cons = 0; 
  	for (int i = 0; i < num_cell; i++) {
  		if (mask[i] == 1) {
  			num_cell_cons += 1;
  		}
  	}
  	std::cout << "Number of considered cells: " << num_cell_cons << endl;
  	std::cout << "Fraction of considered cells: " 
  	    << ((float)num_cell_cons / (float)num_cell * 100.0) << " %" << endl;

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
    vector<geom_vector> north_direct = north_direction(circumcenters, num_cell,
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
    ecef2enu_point(circumcenters, lon_orig, lat_orig, rad_earth);
    ecef2enu_point(vertices, lon_orig, lat_orig, rad_earth);
    ecef2enu_vector(sphere_normals, lon_orig, lat_orig);
    ecef2enu_vector(north_direct, lon_orig, lat_orig);

    // ------------------------------------------------------------------------
    // Initialisation of BVH
    // ------------------------------------------------------------------------

    // Create vertex buffer (of 32 bit floats) needed for the rtc scene
	vector<float> vertex_buffer(num_vertex * 3);
	int ind = 0;
	std::cout << "num_vertex: " << num_vertex << endl;
	for (int i = 0; i < num_vertex; i++){
	    vertex_buffer[ind + 0] = (float)vertices[i].x;
	    vertex_buffer[ind + 1] = (float)vertices[i].y;
	    vertex_buffer[ind + 2] = (float)vertices[i].z;
	    ind += 3;
	}
	std::cout << "ind:" << ind << endl;
	vertex_buffer = pad_buffer(vertex_buffer);





	std::cout << "Exit program!" << endl;
	return;

    

    
    
	// create the vector with vertices and the indices
	// for the creation of the RTC buffer
	vector<int> vertex_of_cell_buffer;
	for(int i = 0; i < num_cell; i++){
		//creation of the buffer of indices needed for the rtc scene
		vertex_of_cell_buffer.push_back(vertex_of_cell[i]);
		vertex_of_cell_buffer.push_back(vertex_of_cell[i + num_cell]);
		vertex_of_cell_buffer.push_back(vertex_of_cell[i + 2*num_cell]);
	} 




	
	
//     int add_elem = 16;
//     if ((sizeof(vertex_of_cell_buffer) % 16) != 0){
//         add_elem += ((16 - (sizeof(vertex_of_cell_buffer) % 16)));
// 	}
// 	for(int i=0; i < add_elem; i++){
//     	vertex_of_cell_buffer.push_back((int)0);	
// 	}


  	// Initialization and time counting
  	auto start_ini = std::chrono::high_resolution_clock::now();

  	RTCDevice device = initializeDevice();
  	RTCScene scene = initializeScene(device, &vertex_buffer[0], &vertex_of_cell_buffer[0], num_vertex, num_cell);

  	auto end_ini = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = end_ini - start_ini;
  	std::cout << "Total initialisation time: " << time.count() << " s" << endl;

    // ------------------------------------------------------------------------
    // Terrain horizon and sky view factor computation
    // ------------------------------------------------------------------------



	// upper  and lower limits for elevation angle [radians]
  	float elev_ang_up_lim = deg2rad(90);  
	float elev_ang_low_lim = deg2rad(-90);


	float hori_acc = deg2rad(hori_acc_deg);
	
	
	// fix angle for azimuth rotation [radians]
	size_t azim_num = nhori;
	float azim_sin = sin((deg2rad(360)/azim_num)); 
	float azim_cos = cos((deg2rad(360)/azim_num));

	// horizon buffer
	vector<float> hori_buffer(azim_num*num_cell, 0.0);
	// Sky View Factor buffer
	vector<float> svf_buffer(num_cell, 0.0);


	size_t num_rays = 0;
  	std::chrono::duration<double> time_ray = std::chrono::seconds(0);
  	std::chrono::duration<double> time_out = std::chrono::seconds(0);

	// --------------------------------------------------------
	// Horizon computation with Intel Embree 
	// --------------------------------------------------------



    	// --------------------------------------------------------------------
  		// Perform ray tracing
    	// --------------------------------------------------------------------

  		auto start_ray = std::chrono::high_resolution_clock::now();

        int idx1, idx2, idx3;
        double dir1_x, dir1_y, dir1_z, dir2_x, dir2_y, dir2_z;
        double d, param;
		for(int i = 0; i < num_cell; i++){
			// define indices of the vertices of the triangle
			// keep in mind that vertex_of_cell values start from 1 not 0
			idx1 = vertex_of_cell[i] - 1;
			idx2 = vertex_of_cell[i + num_cell] - 1;
			idx3 = vertex_of_cell[i + 2 * num_cell] - 1;

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

			double abs_val = sqrt(norm_x * norm_x + norm_y * norm_y + norm_z * norm_z);

			norm_x = norm_x/abs_val;
			norm_y = norm_y/abs_val;
			norm_z = norm_z/abs_val;
			
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
			std::cout << "Circum " << i << " (" << circumcenters[i].x << ", " << circumcenters[i].y << ", " << circumcenters[i].z << ")   " ; 

			// Transfor the Norm and North vectors from double to float
			float vec_norm_x = (float)sphere_normals[i].x;
			float vec_norm_y = (float)sphere_normals[i].y;
			float vec_norm_z = (float)sphere_normals[i].z;
			std::cout << "Norm " << i << " (" << vec_norm_x << ", " << vec_norm_y << ", " << vec_norm_z << ")   " ;

			float vec_north_x = (float)north_direct[i].x;
			float vec_north_y = (float)north_direct[i].y;
			float vec_north_z = (float)north_direct[i].z;      
			std::cout << "North " << i << " (" << vec_north_x << ", " << vec_north_y << ", " << vec_north_z << ")   " << endl;

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

					svf_buffer[i] = _sky_view_factor(azim_num, hori_buffer, i,
										norm_x, norm_y, norm_z, svf_type);
					
					std::cout << "SVF num_cell nr." << i << ": " << svf_buffer[i] << endl << endl;

			}else{

			}
		
		}

		/*std::cout << "Horizon buffer: " << endl;
		for(int i=0; i<(azim_num*num_cell); i++){
			std::cout << hori_buffer[i] << ", ";
		}
		std::cout<<endl;*/

  		auto end_ray = std::chrono::high_resolution_clock::now();
  		time_ray += (end_ray - start_ray);

    return ;
}