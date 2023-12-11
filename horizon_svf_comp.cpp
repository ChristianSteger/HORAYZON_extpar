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
		vector of points (x, y, z) [m]*/

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
    
    /* Compute unit vectors pointing towards North in earth-centered,
    earth-fixed (ECEF) coordinates. These vectors are perpendicular to surface
    normal unit vectors.*/

    geom_vector v_p = {0.0, 0.0, rad_earth};  // north pole in ECEF coordinates
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

    /* In-place coordinate transformation of points from ECEF to ENU coordinate
    system.*/

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

    /*In-place coordinate transformation of vectors from ECEF to ENU coordinate
    system.*/

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
  	//ray.tfar = std::numeric_limits<float>::infinity();
  	ray.tfar = dist_search;

  	// Intersect ray with scene - function that checks
	// wheter there is a hit with the scene
  	rtcOccluded1(scene, &context, &ray);
  
  	return (ray.tfar < 0.0);
}

//-----------------------------------------------------------------------------
// Cast single ray (intersect1; lower performance)
//-----------------------------------------------------------------------------

bool castRay_intersect1(RTCScene scene, float ox, float oy, float oz, float dx,
	float dy, float dz, float dist_search, float &dist) {
  
	// Intersect context
	struct RTCIntersectContext context;
	rtcInitIntersectContext(&context);

  	// Ray hit structure
  	struct RTCRayHit rayhit;
  	rayhit.ray.org_x = ox;
  	rayhit.ray.org_y = oy;
  	rayhit.ray.org_z = oz;
  	rayhit.ray.dir_x = dx;
  	rayhit.ray.dir_y = dy;
  	rayhit.ray.dir_z = dz;
  	rayhit.ray.tnear = 0.0;
  	//rayhit.ray.tfar = std::numeric_limits<float>::infinity();
  	rayhit.ray.tfar = dist_search;
  	//rayhit.ray.mask = -1;
  	//rayhit.ray.flags = 0;
  	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  	// Intersect ray with scene
  	rtcIntersect1(scene, &context, &rayhit);
  	dist = rayhit.ray.tfar;

  	return (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID);

}

//#############################################################################
// Horizon detection algorithms (horizon elevation angle)
//#############################################################################

//-----------------------------------------------------------------------------
// Guess horizon from previous azimuth direction
//-----------------------------------------------------------------------------

void ray_guess_const(float ray_org_x, float ray_org_y, float ray_org_z,
	size_t azim_num, int refine_factor, float hori_acc, float dist_search,
	float elev_ang_low_lim, float elev_ang_up_lim,
	RTCScene scene, size_t &num_rays, vector<float>& hori_buffer,
	float norm_x, float norm_y, float norm_z,
	float north_x, float north_y, float north_z,
	float azim_sin, float azim_cos, int num_cell) {

	// ------------------------------------------------------------------------
  	// First azimuth direction (binary search - faster)
  	// ------------------------------------------------------------------------  
	
	//vector of horizon values before refining
	vector <float> hori_not_averaged((int)azim_num*refine_factor);
	float param;
  	float lim_up = elev_ang_up_lim;
  	float lim_low = elev_ang_low_lim;

	// elev_ang in my case is the angle that defines
	// the rotation angle for Rodrigues' formula
  	float elev_ang = (abs(lim_up) - abs(lim_low)) / 2.0;
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
	float dir_x, dir_y, dir_z;
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
		
		dir_x = new_dir_x;
		dir_y = new_dir_y;
		dir_z = new_dir_z;

		new_dir_x = dir_x * elev_cos + 
			(counterclock_prod_y * dir_z - counterclock_prod_z * dir_y) * elev_sin + 
			counterclock_prod_x * param;
		new_dir_y = new_dir_y * elev_cos + 
			(counterclock_prod_z * dir_x - counterclock_prod_x * dir_z) * elev_sin + 
			counterclock_prod_y * param;
		new_dir_z = dir_z * elev_cos + 
			(counterclock_prod_x * dir_y - counterclock_prod_y * dir_x) * elev_sin + 
			counterclock_prod_z * param;
		
		final_ang += elev_ang;	 	
  	}

	
	if (rad2deg(final_ang) >= 0){
  		hori_buffer[azim_num * num_cell] = final_ang;
	} else {
		hori_buffer[azim_num * num_cell] = 0.0;	
	}   
	//hori_not_averaged[0] = final_ang;
	//std::cout << "Horizon first azimuth: " << rad2deg(final_ang) << endl;
	//std::cout << "Horizon not averaged:" << "[0: " << rad2deg(hori_not_averaged[0]) << ",";

	// ------------------------------------------------------------------------
	// Remaining azimuth directions (guess horizon from previous
	// azimuth direction)
	// ------------------------------------------------------------------------

	//for (size_t k = 1; k < refine_factor*azim_num; k++){ //sampling case
	for (size_t k = 1; k < azim_num; k++){

		// Rodrigues' rotation formula TO CHANGE THE AZIMUTH
		// the norm_ vector is the rotation axis (so we'll rotate counter clockwise);  
		// the vector that must be rotated is the previous direction
 		float param = (1 - azim_cos) * 
			(norm_x * new_dir_x + norm_y * new_dir_y + norm_z * new_dir_z);

		dir_x = new_dir_x;
		dir_y = new_dir_y;
		dir_z = new_dir_z;
		
		new_dir_x = dir_x * azim_cos + 
			(norm_y * dir_z - norm_z * dir_y) * azim_sin + 
			norm_x * param;
		new_dir_y = dir_y * azim_cos + 
			(norm_z * dir_x - norm_x * dir_z) * azim_sin + 
			norm_y * param;
		new_dir_z = dir_z * azim_cos + 
			(norm_x * dir_y - norm_y * dir_x) * azim_sin + 
			norm_z * param; 

		// now focus on elevation rotation
 		param = (1 - azim_cos) * 
			(norm_x * counterclock_prod_x + norm_y * counterclock_prod_y + norm_z * counterclock_prod_z);

		dir_x = counterclock_prod_x;
		dir_y = counterclock_prod_y;
		dir_z = counterclock_prod_z;

		counterclock_prod_x = dir_x * azim_cos + 
			(norm_y * dir_z - norm_z * dir_y) * azim_sin + 
			norm_x * param;
		counterclock_prod_y = dir_y * azim_cos + 
			(norm_z * dir_x - norm_x * dir_z) * azim_sin + 
			norm_y * param;
		counterclock_prod_z = dir_z * azim_cos + 
			(norm_x * dir_y - norm_y * dir_x) * azim_sin + 
			norm_z * param; 		

		// Move upwards to check if the horizon is higher
		double delta = deg2rad(1); // 1 degree in [radians]
		double sin_delta = sin(delta);
		double cos_delta = cos(delta);
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
			
			dir_x = new_dir_x;
			dir_y = new_dir_y;
			dir_z = new_dir_z;

			new_dir_x = dir_x * cos_delta + 
				(counterclock_prod_y * dir_z - counterclock_prod_z * dir_y) * sin_delta + 
				counterclock_prod_x * param;
			new_dir_y = dir_y * cos_delta + 
				(counterclock_prod_z * dir_x - counterclock_prod_x * dir_z) * sin_delta + 
				counterclock_prod_y * param;
			new_dir_z = dir_z * cos_delta + 
				(counterclock_prod_x * dir_y - counterclock_prod_y * dir_x) * sin_delta + 
				counterclock_prod_z * param;

  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);

  			num_rays += 1;
			// increase the count for number of hitting 
			// starting from the previous elevation angle
  			count += 1;		
		}

		// discrete ray sampling until it hits the topography
		while (!hit) {
			final_ang -= delta;	

			// Rodrigues' rotation formula
			param = (1 - cos_delta) * 
				(counterclock_prod_x * new_dir_x + counterclock_prod_y * new_dir_y + counterclock_prod_z * new_dir_z);

			dir_x = new_dir_x;
			dir_y = new_dir_y;
			dir_z = new_dir_z;

			new_dir_x = dir_x * cos_delta + 
				(counterclock_prod_y * dir_z - counterclock_prod_z * dir_y) * (- sin_delta) + 
				counterclock_prod_x * param;
			new_dir_y = dir_y * cos_delta + 
				(counterclock_prod_z * dir_x - counterclock_prod_x * dir_z) * (- sin_delta) + 
				counterclock_prod_y * param;
			new_dir_z = dir_z * cos_delta + 
				(counterclock_prod_x * dir_y - counterclock_prod_y * dir_x) * (- sin_delta) + 
				counterclock_prod_z * param; 
				
  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);

  			num_rays += 1;	
			count +=1;
		}

   	 	if (rad2deg(final_ang) >= 0){
 			hori_buffer[azim_num * num_cell + k] = final_ang;
		} else { 
			hori_buffer[azim_num * num_cell + k] = 0.0; 
		}   
		//hori_not_averaged[k] = final_ang; 
		//std::cout << " " << k << ": " << rad2deg(hori_not_averaged[k]) << ",";
	}

	// for the refined case
	/* for(int k = 0; k < (int)azim_num; k++){

		for(int j = 0; j < refine_factor; j++){
			hori_buffer[num_cell * azim_num + k] += hori_not_averaged[refine_factor*k + j];
		}

		hori_buffer[num_cell * azim_num + k] /= refine_factor;
		if(rad2deg(hori_buffer[num_cell * azim_num + k]) < 0.0){
			hori_buffer[num_cell * azim_num + k] = 0.0;
		}
	} */ 

}


//-----------------------------------------------------------------------------
// Declare function pointer and assign function
//-----------------------------------------------------------------------------

float (*function_pointer)(int azim_num, vector<float> hori_buffer, 
						int num_cell, float vec_norm_x, float vec_norm_y, 
						float vec_norm_z);

//#############################################################################
// Sky View Factor computation algorithms
//#############################################################################

//-----------------------------------------------------------------------------
// pure geometric skyview-factor
//-----------------------------------------------------------------------------

float pure_geometric_svf(int azim_num, vector<float> hori_buffer, int num_cell,
                        float vec_norm_x, float vec_norm_y, float vec_norm_z){

	float svf = 0;
	for(int k = 0; k < azim_num; k++){
		svf += (1 - sin(hori_buffer[num_cell * azim_num + k]));	
	}
	svf /= azim_num;

    return svf;
}

//-----------------------------------------------------------------------------
// skyview-factor computation scaled with sin(horizon)
//-----------------------------------------------------------------------------
float geometric_svf_scaled_1(int azim_num, vector<float> hori_buffer, 
							int num_cell, float vec_norm_x, float vec_norm_y,
							float vec_norm_z){

	float svf = 0;
	for(int k = 0; k < azim_num; k++){
		svf += (1 - (sin(hori_buffer[num_cell * azim_num + k]) 
				* sin(hori_buffer[num_cell * azim_num + k])));
	}
	svf /= azim_num;

	return svf;
}

//-----------------------------------------------------------------------------
// skyview-factor computation scaled with sin(horizon)**2
//-----------------------------------------------------------------------------
float geometric_svf_scaled_2(int azim_num, vector<float> hori_buffer, 
							int num_cell, float vec_norm_x, float vec_norm_y,
							float vec_norm_z){

	float svf = 0;
	for(int k = 0; k < azim_num; k++){
		svf += (1 - (sin(hori_buffer[num_cell * azim_num + k]) 
					* sin(hori_buffer[num_cell * azim_num + k]) 
					* sin(hori_buffer[num_cell * azim_num + k])));

	}
	svf /= azim_num;

	return svf;
}

//-----------------------------------------------------------------------------
// skyview-factor for sloped surface according to HORAYZON
//-----------------------------------------------------------------------------
float sky_view_factor(int azim_num, vector<float> hori_buffer, 
							int num_cell, float vec_norm_x, float vec_norm_y,
							float vec_norm_z){

	float svf = 0;
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

	return svf;
}

//#############################################################################
// Main function
//#############################################################################

void horizon_svf_comp(double* vlon, double* vlat, float* topography_v,
	int num_vertex,
	double* clon, double* clat,  int* vertex_of_cell,
    int num_cell,
	float* horizon, float* skyview, int nhori, int refine_factor, int svf_type){

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
	/* ecef2enu_point(circumcenters, lon_orig, lat_orig, rad_earth);
    ecef2enu_point(vertices, lon_orig, lat_orig, rad_earth);
    ecef2enu_vector(sphere_normals, lon_orig, lat_orig);
    ecef2enu_vector(north_direct, lon_orig, lat_orig);  */

	// ------------------------------------------------------------------------
    // Building of BVH
    // ------------------------------------------------------------------------

  	RTCDevice device = initializeDevice();
	RTCScene scene = initializeScene(device, vertex_of_cell, num_cell,
  	    vertices);

    // ------------------------------------------------------------------------
    // Test ray casting --------------------------------------------------------> remove again later...
    // ------------------------------------------------------------------------
/*    
    for (int i = 0; i < num_cell; i++){
    
        float ray_orig_x = circumcenters[i].x;
        float ray_orig_y = circumcenters[i].y;
        float ray_orig_z = circumcenters[i].z;
        float ray_dir_x = -sphere_normals[i].x;
        float ray_dir_y = -sphere_normals[i].y;
        float ray_dir_z = -sphere_normals[i].z;
        float dist = 0.0;

        bool hit = castRay_intersect1(scene, ray_orig_x, ray_orig_y,
            ray_orig_z, ray_dir_x, ray_dir_y, ray_dir_z, dist_search, dist);
        std::cout << "hit: " << hit << endl;
        std::cout << "distance to hit: " << dist << " m" << endl;
    
    }

	std::cout << "Exit program!" << endl;
	return;
*/	
    // ------------------------------------------------------------------------
    // Terrain horizon and sky view factor computation
    // ------------------------------------------------------------------------

	// upper  and lower limits for elevation angle [radians]
  	float elev_ang_up_lim = deg2rad(90);  
	float elev_ang_low_lim = deg2rad(-90);

	// horizon accuracy [degree] --> [radians]
	float hori_acc = deg2rad(hori_acc_deg);

	
	// fix angle for azimuth rotation [radians]
	size_t azim_num = nhori;
	float azim_sin = sin((deg2rad(360)/azim_num)); 
	float azim_cos = cos((deg2rad(360)/azim_num));

	// horizon buffer
	vector<float> hori_buffer(azim_num*num_cell, 0.0);
	// Sky View Factor buffer
	vector<float> svf_buffer(num_cell, 0.0);

	// Select algorithm for sky view factor computation
	std::cout << "Sky View Factor computation algorithm: ";
	if (svf_type == 1) {
		std::cout << "pure geometric svf." << endl;		
		function_pointer = pure_geometric_svf;
	} else if (svf_type == 2) {
		std::cout << "geometric scaled with sin(horizon)." << endl;
		function_pointer = geometric_svf_scaled_1;
	} else if (svf_type == 3) {
		std::cout << "geometric scaled with sin(horizon)**2." << endl;
		function_pointer = geometric_svf_scaled_2;
	} else if (svf_type == 4){
		std::cout << "SVF for sloped surface according to HORAYZON." << endl;
		function_pointer = sky_view_factor;		
	}

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
		idx1 = vertex_of_cell[i];
		idx2 = vertex_of_cell[i + num_cell];
		idx3 = vertex_of_cell[i + 2 * num_cell];

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

		// Transfor the Norm and North vectors from double to float
		float vec_norm_x = (float)sphere_normals[i].x;
		float vec_norm_y = (float)sphere_normals[i].y;
		float vec_norm_z = (float)sphere_normals[i].z;

		float vec_north_x = (float)north_direct[i].x;
		float vec_north_y = (float)north_direct[i].y;
		float vec_north_z = (float)north_direct[i].z;     

		// circumcenters from double to float
		float circ_x = (float)circumcenters[i].x;
		float circ_y = (float)circumcenters[i].y;
		float circ_z = (float)circumcenters[i].z;  

		// Ray-tracing origin	
		float ray_org_x = circ_x 
			+ vec_norm_x * ray_org_elev;
		float ray_org_y = circ_y 
			+ vec_norm_y * ray_org_elev;
		float ray_org_z = circ_z 
			+ vec_norm_z * ray_org_elev;

		ray_guess_const(ray_org_x, ray_org_y, ray_org_z, azim_num, refine_factor,
			hori_acc, dist_search, elev_ang_low_lim, elev_ang_up_lim, 
			scene, num_rays, hori_buffer, vec_norm_x, 
			vec_norm_y, vec_norm_z, vec_north_x, 
			vec_north_y, vec_north_z, azim_sin, azim_cos, i);	

		svf_buffer[i] = function_pointer(azim_num, hori_buffer, i,
							norm_x, norm_y, norm_z);
		
	}


  	/* std::cout << "Horizon buffer: " << endl;
	//for(int i=0; i<((int)azim_num*num_cell); i++){
	for(int i=0; i<(int)azim_num; i++){
		std::cout << rad2deg(hori_buffer[i]) << ", ";
	}
	std::cout<<endl<<endl;  
	auto min_elem = std::min_element(hori_buffer.begin(), hori_buffer.end());
	auto max_elem = std::max_element(hori_buffer.begin(), hori_buffer.end());
	std::cout << "Min: " << rad2deg(*min_elem) << "   Max:" << rad2deg(*max_elem) << endl;  */

	/* std::cout << "SVF: " << endl;
	for(int i=0; i<num_cell; i++){
	//for(int i=0; i<1; i++){
		std::cout << svf_buffer[i] << ", ";
	}
	std::cout<<endl; */   

	auto end_ray = std::chrono::high_resolution_clock::now();
	time_ray += (end_ray - start_ray);

    return ;
}