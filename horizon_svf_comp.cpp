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


// STRUCTURE FOR (x,y,z) COORDINATES
struct coordinates{
    vector<double> x;
    vector<double> y;
    vector<double> z;
};
typedef struct coordinates coords;

// STRUCTURE FOR (x,y,z) COORDINATES
struct plane{
    vector<double> a;
    vector<double> b;
    vector<double> c;
    vector<double> d;
};
typedef struct plane plane_cartesian;

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

coords lonlat2ecef(double* vlon, double* vlat, double* h, int len){

    /*Coordinate transformation from lon/lat to ECEF.

    Transformation of geodetic longitude/latitude to earth-centered,
    earth-fixed (ECEF) coordinates.

    Parameters
    ----------
    vlon : ndarray of double
        Array (with arbitrary dimensions) with geographic longitude [degree]
    vlat : ndarray of double
        Array (with arbitrary dimensions) with geographic latitude [degree]
    h : ndarray of float
        Array (with arbitrary dimensions) with elevation above ellipsoid
        [metre]

    Returns
    -------
    x_ecef : ndarray of double
        Array (dimensions according to input) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (dimensions according to input) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (dimensions according to input) with ECEF z-coordinates [metre]
        
    Sources
    -------
    - https://en.wikipedia.org/wiki/Geographic_coordinate_conversion
    - Geoid parameters r, a and f: PROJ */

    // initialization
    coords xyz_ecef;
    vector<double> x_ecef(len);
    vector<double> y_ecef(len);
    vector<double> z_ecef(len);

    // Spherical coordinates
    double r = 6370997.0;  // earth radius [m]

    // before it was 
    // for i in prange(len_0, nogil=True, schedule="static")
    // in which prange is to run in parallel in cython (don't know how to change it in c++)
    for (int i = 0; i < len; i++){
        x_ecef[i] = (r + h[i]) * cos(vlat[i]) * cos(vlon[i]);
        y_ecef[i] = (r + h[i]) * cos(vlat[i]) * sin(vlon[i]);
        z_ecef[i] = (r + h[i]) * sin(vlat[i]);
    }

    xyz_ecef.x = x_ecef;
    xyz_ecef.y = y_ecef;
    xyz_ecef.z = z_ecef;

    //xyz_ecef = _lonlat2ecef_1d(vlon, vlat, h, len);
    return xyz_ecef;
}

// -----------------------------------------------------------------------------

coords ecef2enu(vector<double>& x_ecef, vector<double>& y_ecef, 
	vector<double>& z_ecef, int len, double lon_or, double lat_or){

    /*Coordinate transformation from ECEF to ENU.

    Transformation of earth-centered, earth-fixed (ECEF) to local tangent
    plane (ENU) coordinates.

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF z-coordinates [metre]

    Returns
    -------
    x_enu : ndarray of float
        Array (dimensions according to input) with ENU x-coordinates [metre]
    y_enu : ndarray of float
        Array (dimensions according to input) with ENU y-coordinates [metre]
    z_enu : ndarray of float
        Array (dimensions according to input) with ENU z-coordinates [metre] */

    // Check arguments
    double sin_lon, cos_lon, sin_lat, cos_lat;
	double x_ecef_or, y_ecef_or, z_ecef_or;
    vector<double> x_enu(len);
    vector<double> y_enu(len);
    vector<double> z_enu(len);
    coords xyz_enu;

    // Check and change values to the latitude and longitude coordinates of the origin
    try{

       /* if(!(is_same<decltype(x_ecef), double>::value) ||
            !(is_same<decltype(y_ecef), double>::value) ||
            !(is_same<decltype(z_ecef), double>::value) ){
            throw invalid_argument("Input array(s) has/have incorrect data type(s)");
        } */     
        if (lon_or < -180.0 || lon_or > 180.0) {
            throw invalid_argument("Value for 'lon_or' is outside of valid range");
        }
        if (lat_or < -90.0 || lat_or > 90.0) {
            throw invalid_argument("Value for 'lat_or' is outside of valid range");
        }
      
    }catch (const invalid_argument& e) {
        cerr << "Error: " << e.what() << endl;
    }

    const double r = 6370997.0; // earth radius [m]
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
        x_enu[i] = (- sin_lon * (x_ecef[i] - x_ecef_or) 
                    + cos_lon * (y_ecef[i] - y_ecef_or));
        y_enu[i] = (- sin_lat * cos_lon * (x_ecef[i] - x_ecef_or)
                    - sin_lat * sin_lon * (y_ecef[i] - y_ecef_or)
                    + cos_lat * (z_ecef[i] - z_ecef_or));
        z_enu[i] = (+ cos_lat * cos_lon * (x_ecef[i] - x_ecef_or)
                    + cos_lat * sin_lon * (y_ecef[i] - y_ecef_or)
                    + sin_lat * (z_ecef[i] - z_ecef_or));
    }
    // it was: return np.asarray(x_enu), np.asarray(y_enu), np.asarray(z_enu)
    // need to change dimensions?
    xyz_enu.x = x_enu;
    xyz_enu.y = y_enu;
    xyz_enu.z = z_enu;

    return xyz_enu;
}



//#############################################################################
// Compute normal unit vector in ECEF coordinates
//#############################################################################

coords surf_norm(double* lon, double* lat, int len){
    /*Compute surface normal unit vectors.

    Computation of surface normal unit vectors in earth-centered, earth-fixed
    (ECEF) coordinates.

    Parameters
    ----------
    lon : ndarray of double
        Array (with arbitrary dimensions) with geographic longitude [degree]
    lat : ndarray of double
        Array (with arbitrary dimensions) with geographic latitudes [degree]

    Returns
    -------
    vec_norm_ecef : ndarray of double
        Array (dimensions according to input; vector components are stored in
        last dimension) with surface normal components in ECEF coordinates
        [metre]  
    
    Sources
    -------
    - https://en.wikipedia.org/wiki/N-vector */

    double sin_lon, cos_lon, sin_lat, cos_lat;
    vector<double> vec_norm_ecef_x(len);
    vector<double> vec_norm_ecef_y(len);
    vector<double> vec_norm_ecef_z(len);
    coords vec_norm;

    // Compute surface normals
    // for i in prange(len_0, nogil=True, schedule="static"):
    for (int i = 0; i < len; i++ ){
        sin_lon = sin(lon[i]);
        cos_lon = cos(lon[i]);
        sin_lat = sin(lat[i]);
        cos_lat = cos(lat[i]);
        vec_norm_ecef_x[i] = cos_lat * cos_lon;
        vec_norm_ecef_y[i] = cos_lat * sin_lon;
        vec_norm_ecef_z[i] = sin_lat;
    }

    vec_norm.x = vec_norm_ecef_x;
    vec_norm.y = vec_norm_ecef_y;
    vec_norm.z = vec_norm_ecef_z;

    return vec_norm;
}

coords north_dir(vector<double> x_ecef, vector<double> y_ecef, vector<double> z_ecef, int len, coords vec_norm_ecef){
    /*Compute unit vectors pointing towards North.

    Computation unit vectors pointing towards North in earth-centered,
    earth-fixed (ECEF) coordinates. These vectors are perpendicular to surface
    normal unit vectors.

    Parameters
    ----------
    x_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF x-coordinates [metre]
    y_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF y-coordinates [metre]
    z_ecef : ndarray of double
        Array (with arbitrary dimensions) with ECEF z-coordinates [metre]
    vec_norm_ecef : ndarray of float
        Array (at least two-dimensional; vector components must be stored in
        last dimension) with surface normal components in ECEF coordinates
        [metre]

    Returns
    -------
    vec_north_ecef : ndarray of float
        Array (dimensions according to input; vector components are stored in
        last dimension) with north vector components in ECEF coordinates
        [metre]*/

    double np_x, np_y, np_z;
    double vec_nor_x, vec_nor_y, vec_nor_z;
    double dot_pr, vec_proj_x, vec_proj_y, vec_proj_z, norm;
	vector<double> north_ecef_x(len), north_ecef_y(len), north_ecef_z(len);
    coords vec_north_ecef;
	double r = 6370997.0;  // earth radius [m]

    // Coordinates of North pole
    np_x = 0.0;
    np_y = 0.0;
    np_z = r;

    // Coordinate transformation
    for (int i = 0; i < len; i++){

        // Vector to North Pole
        vec_nor_x = (np_x - x_ecef[i]);
        vec_nor_y = (np_y - y_ecef[i]);
        vec_nor_z = (np_z - z_ecef[i]);

		/*std::cout << "Vector to North Pole: (" << vec_nor_x << ", " << vec_nor_y << ", " 
			<< vec_nor_z << ");" << endl;*/


        // Project vector to North Pole on surface normal plane
        dot_pr = ((vec_nor_x * vec_norm_ecef.x[i])
                  + (vec_nor_y * vec_norm_ecef.y[i])
                  + (vec_nor_z * vec_norm_ecef.z[i]));
		
		/*std::cout << "  Scalar product: " << dot_pr << ";" << endl;*/

        vec_proj_x = vec_nor_x - dot_pr * vec_norm_ecef.x[i];
        vec_proj_y = vec_nor_y - dot_pr * vec_norm_ecef.y[i];
        vec_proj_z = vec_nor_z - dot_pr * vec_norm_ecef.z[i];

		/*std::cout << "Vector projection on surface normal plane: (" 
			<< vec_proj_x << ", " << vec_proj_y << ", " << vec_proj_z << ");" << endl;*/

        // Normalise vector
        norm = sqrt(vec_proj_x * vec_proj_x + vec_proj_y * vec_proj_y + vec_proj_z * vec_proj_z);

		/*std::cout << "Norm: " << norm << endl;*/

		north_ecef_x[i] = vec_proj_x / norm;
        north_ecef_y[i] = vec_proj_y / norm;
        north_ecef_z[i] = vec_proj_z / norm;

	}
	
	vec_north_ecef.x = north_ecef_x;
	vec_north_ecef.y = north_ecef_y;
	vec_north_ecef.z = north_ecef_z;	

    return vec_north_ecef;
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
	//float* vert_simp, int num_vert_simp, int* tri_ind_simp, int num_tri_simp) {

	RTCScene scene = rtcNewScene(device);
  	rtcSetSceneFlags(scene, RTC_SCENE_FLAG_ROBUST);

  	RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);  	
  	rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0,
  		RTC_FORMAT_FLOAT3, vert_grid, 0, 3*sizeof(float), num_vert);  	
	
	// assign a INDEX data buffer to the geometry
    Triangle* triangles = (Triangle*) rtcSetNewGeometryBuffer(geom,
        RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Triangle),
        num_tri);

	for (int i = 0; i < num_tri; i++) {
			triangles[i].v0 = vertex_of_cell[3 * i]-1;
			triangles[i].v1 = vertex_of_cell[3 * i + 1]-1;
			triangles[i].v2 = vertex_of_cell[3 * i + 2]-1;

			std::cout << "triangle nr. " << i << " : (" << triangles[i].v0 << ", " << triangles[i].v1 << ", " << triangles[i].v2 << ");  ";
	}
	std::cout << endl;

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
  	//ray.tfar = std::numeric_limits<float>::infinity();
  	ray.tfar = dist_search;
  	//ray.mask = -1;
  	//ray.flags = 0;

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
	float azim_sin, float azim_cos) {

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

	float clock_prod_x, clock_prod_y, clock_prod_z;
	float counterclock_prod_x, counterclock_prod_y, counterclock_prod_z;

	// cross product to then have the normal axis 
	// that allows a counter-clockwise rotation
	counterclock_prod_x = north_y * norm_z - north_z * norm_y;
	counterclock_prod_y = north_z * norm_x - north_x * norm_z;
	counterclock_prod_z = north_x * norm_y - north_y * norm_x;

	// cross product to then have the normal axis 
	// that allows a counter-clockwise rotation
	clock_prod_x = - counterclock_prod_x;
	clock_prod_y = - counterclock_prod_y;
	clock_prod_z = - counterclock_prod_z;

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


			elev_sin = sin(abs(elev_ang));
			elev_cos = cos(abs(elev_ang));

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

  		} else {
			hit_num++;
  			lim_up = final_ang;		
			elev_ang = (abs(lim_up) - abs(lim_low)) / 2.0;

			elev_sin = sin(abs(elev_ang));
			elev_cos = cos(abs(elev_ang));

			// Rodrigues' rotation formula
			param = (1 - elev_cos) * 
				(clock_prod_x * new_dir_x + clock_prod_y * new_dir_y + clock_prod_z * new_dir_z);

			new_dir_x = new_dir_x * elev_cos + 
				(clock_prod_y * new_dir_z - clock_prod_z * new_dir_y) * elev_sin + 
				clock_prod_x * param;
			new_dir_y = new_dir_y * elev_cos + 
				(clock_prod_z * new_dir_x - clock_prod_x * new_dir_z) * elev_sin + 
				clock_prod_y * param;
			new_dir_z = new_dir_z * elev_cos + 
				(clock_prod_x * new_dir_y - clock_prod_y * new_dir_x) * elev_sin + 
				clock_prod_z * param;
			
			final_ang += elev_ang;

  		}
		
		if ((hit_num==1) || (hit_num == 2) || (hit_num == 3) || (hit_num == 4) || (hit_num == 5) || (hit_num == 6)){
			std::cout << "lim_up = " << rad2deg(lim_up) << endl;
			std::cout << "lim_low = " << rad2deg(lim_low) << endl;
			std::cout << "elev_ang = " << rad2deg(elev_ang) << endl;
			std::cout << "final_ang = " << rad2deg(final_ang) << endl; 
		}
  				
  	}

  	hori_buffer[0] = final_ang;
	std::cout << "Horizon first azimuth: " << rad2deg(hori_buffer[0]) << endl;
	// initialize the variable containing the previous azimuth value
  	float prev_elev_ang = final_ang;

	// ------------------------------------------------------------------------
	// Remaining azimuth directions (guess horizon from previous
	// azimuth direction)
	// ------------------------------------------------------------------------
/*	
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

		// cross product to then have a clockwise rotation
		clock_prod_x = - counterclock_prod_x;
		clock_prod_y = - counterclock_prod_y;
		clock_prod_z = - counterclock_prod_z;

		// Move upwards to check if the horizon is higher
		int delta = 0.15; //0.0175; // [radians]
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

  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);
				
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
				(clock_prod_x * new_dir_x + clock_prod_y * new_dir_y + clock_prod_z * new_dir_z);

			new_dir_x = new_dir_x * cos_delta + 
				(clock_prod_y * new_dir_z - clock_prod_z * new_dir_y) * sin_delta + 
				clock_prod_x * param;
			new_dir_y = new_dir_y * cos_delta + 
				(clock_prod_z * new_dir_x - clock_prod_x * new_dir_z) * sin_delta + 
				clock_prod_y * param;
			new_dir_z = new_dir_z * cos_delta + 
				(clock_prod_x * new_dir_y - clock_prod_y * new_dir_x) * sin_delta + 
				clock_prod_z * param;

  			hit = castRay_occluded1(scene, ray_org_x, ray_org_y,
  				ray_org_z, new_dir_x, new_dir_y, new_dir_z,
  				dist_search);
  			num_rays += 1;		
		}

  		hori_buffer[k] = final_ang;
		std::cout << "Horizon azimuth nr." << k << ": " << elev_ang << endl;
	}
*/
}

//#############################################################################
// MAIN FUNCTION
//#############################################################################


void horizon_svf_comp(double* vlon, double* vlat, double* clon, 
    double* clat, float* topography_v, int* vertex_of_cell, 
    int vertex, int cell, float* horizon, float* skyview, int nhori, int* mask){
    
    int nv = 3;  // number of vertices per cell

    std::cout << "--------------------------------------------------------" << endl;
	std::cout << "Horizon and Sky View Factor(SVF) computation with Intel Embree" << endl;
	std::cout << "--------------------------------------------------------" << endl;

    coords circ_ecef, vert_ecef, vert_enu, circ_enu, v1, v2, v3;
    int idx1, idx2, idx3;
    // fixing elevation to 0.0
    vector<double> h_v(vertex, 0.0);
    vector<double> h_c(cell, 0.0); 

    // initialize the vectors containing 
    // components of the normal vector for each triangle and of the
    // points of intersection between each triangle and the circumcenter
    vector<double> norm_x(cell);
    vector<double> norm_y(cell);
    vector<double> norm_z(cell);
    vector<double> intersect_x(cell);
    vector<double> intersect_y(cell);
    vector<double> intersect_z(cell);

    double dir1_x, dir1_y, dir1_z, dir2_x, dir2_y, dir2_z;
    double d, param;

    circ_ecef = lonlat2ecef(clon, clat, &h_c[0], cell);
    vert_ecef = lonlat2ecef(vlon, vlat, &h_v[0], vertex);
	/*for (int i = 0; i < vertex; i++){
		std::cout << "(" << vert_ecef.x[i] << ", " << vert_ecef.y[i] << ", " << vert_ecef.z[i] << ")" << " ";
	}
	std::cout << endl;
	for (int i = 0; i < cell; i++){
		std::cout << "(" << circ_ecef.x[i] << ", " << circ_ecef.y[i] << ", " << circ_ecef.z[i] << ")" << " ";
	}
	std::cout << endl;*/

	/*for (int i = 0; i < 3*cell; i++){
		std::cout << vertex_of_cell[i] << ", ";
	}
	std::cout << endl;*/

	vector<int> vertex_of_cell_buffer;

    for(int i = 0; i < cell; i++){
	//for(int i = 0; i < 2; i++){
		// define indices of the vertices of the triangle
		// keep in mind that vertex_of_cell values start from 1 not 0
		idx1 = vertex_of_cell[i] - 1;
		idx2 = vertex_of_cell[i + cell] - 1;
		idx3 = vertex_of_cell[i + 2 * cell] - 1;

		//creation of the buffer of indices needed for the rtc scene
		vertex_of_cell_buffer.push_back(vertex_of_cell[i]);
		vertex_of_cell_buffer.push_back(vertex_of_cell[i + cell]);
		vertex_of_cell_buffer.push_back(vertex_of_cell[i + 2*cell]);

		//compute two directions that will allow 
		// to generate the plane of the cell
        dir1_x = vert_ecef.x[idx2] - vert_ecef.x[idx1];
        dir1_y = vert_ecef.y[idx2] - vert_ecef.y[idx1];
        dir1_z = vert_ecef.z[idx2] - vert_ecef.z[idx1];
        dir2_x = vert_ecef.x[idx3] - vert_ecef.x[idx1];
        dir2_y = vert_ecef.y[idx3] - vert_ecef.y[idx1];
        dir2_z = vert_ecef.z[idx3] - vert_ecef.z[idx1];

        // compute coeffiecients of the normal of the trg
		norm_x[i] = dir1_y * dir2_z - dir1_z * dir2_y;
    	norm_y[i] = dir1_z * dir2_x - dir1_x * dir2_z;
    	norm_z[i] = dir1_x * dir2_y - dir1_y * dir2_x;

        // compute the constant term of the plane 
        d = - norm_x[i]*vert_ecef.x[idx1] - norm_y[i]*vert_ecef.y[idx1] - norm_z[i]*vert_ecef.z[idx1];

        // compute intersection between the line and the plane
        // note that in our case, since the origin is (0,0,0)
        // the line that we have to intersect is simply defined 
        // by the coordinates of the circumcenter 
		param = - d / (norm_x[i]*circ_ecef.x[i] + norm_y[i]*circ_ecef.y[i] + norm_z[i]*circ_ecef.z[i]);
 
        intersect_x[i] = param * circ_ecef.x[i]; 
        intersect_y[i] = param * circ_ecef.y[i]; 
        intersect_z[i] = param * circ_ecef.z[i]; 
        
    }


	// --------------------------------------------------------
	// Horizon computation with Intel Embree 
	// --------------------------------------------------------

  	// Initialization and time counting
  	auto start_ini = std::chrono::high_resolution_clock::now();

	// create the vector with vertices [(x,y,z), .. , (x,y,z)]
	// for the creation of the RTC buffer
	vector<float> vert_buffer;	
	//vector<float> lonlat_buffer;

	// if I want to change to ENU before
	double lon_or=vlon[int(vertex / 2)];
	double lat_or=vlat[int(vertex / 2)];
	std::cout << "Origin:" << endl << "longitude: " << lon_or << ", latitude: " << lat_or << endl;

	vert_enu = ecef2enu(vert_ecef.x, vert_ecef.y, vert_ecef.z, vertex, lon_or, lat_or);
	circ_enu = ecef2enu(circ_ecef.x, circ_ecef.y, circ_ecef.z, cell, lon_or, lat_or);
	/*for (int i = 0; i < vertex; i++){
		std::cout << "(" << vert_enu.x[i] << ", " << vert_enu.y[i] << ", " << vert_enu.z[i] << ")" << " ";
	}
	std::cout << endl;*/

	for (int i = 0; i < vertex; i++){
		//convert double elements into floats and then add the element to vert_grid
		vert_buffer.push_back((float)vert_ecef.x[i]);
		vert_buffer.push_back((float)vert_ecef.y[i]);
		vert_buffer.push_back((float)vert_ecef.z[i]);

		/*lonlat_buffer.push_back((float)vlon[i]);
		lonlat_buffer.push_back((float)vlat[i]);*/

		/*vert_buffer.push_back((float)vert_enu.x[i]);
		vert_buffer.push_back((float)vert_enu.y[i]);
		vert_buffer.push_back((float)vert_enu.z[i]);*/
	}
	//pad-buffer function
	vert_buffer = pad_buffer(vert_buffer);
/*
	// vertex_of_cell_buffer
    int add_elem = 16;
    if ((sizeof(vertex_of_cell_buffer) % 16) != 0){
        add_elem += ((16 - (sizeof(vertex_of_cell_buffer) % 16)));
	}
	for(int i=0; i < add_elem; i++){
    	vertex_of_cell_buffer.push_back((int)0);	
	}
*/
	/*
	for (int i = 0; i < 3*cell; i++){
		std::cout << vertex_of_cell_buffer[i] << ", ";
	}
	std::cout << endl;*/

  	RTCDevice device = initializeDevice();
  	RTCScene scene = initializeScene(device, &vert_buffer[0], &vertex_of_cell_buffer[0], vertex, cell);


  	auto end_ini = std::chrono::high_resolution_clock::now();
  	std::chrono::duration<double> time = end_ini - start_ini;
  	//std::cout << "Total initialisation time: " << time.count() << " s" << endl;

	// upper  and lower limits for elevation angle [degree]
  	float elev_ang_up_lim_deg = 90.0;  
	float elev_ang_low_lim_deg = -90.0;	
	// in RADIANS
  	float elev_ang_up_lim = deg2rad(90.0);  
	float elev_ang_low_lim = deg2rad(-90.0);

	float hori_acc_deg = 15; // horizon accuracy in radians
	float hori_acc = deg2rad(hori_acc_deg);
	float dist_search = 20;  //[kilometer]
  	dist_search *= 1000.0;  // [kilometer] -> [metre]

	// number of azimuth sectors
	float hori_buffer_size_max = 1.5;	

	// fix elevation offset
	float ray_org_elev=0.01;
	// fix angle for azimuth rotation (RADIANS)
	size_t azim_num = nhori;
	float azim_sin = sin((deg2rad(360)/azim_num)); 
	float azim_cos = cos((deg2rad(360)/azim_num));

	// in DEGREES
	float azim_sin_deg = sin(360/azim_num); 
	float azim_cos_deg = cos(360/azim_num);

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

  	// Compute normals to the surface and north vectors in the circumcenters
	coords vec_norm_ecef, vec_north_ecef;
	vec_norm_ecef = surf_norm(clon, clat, cell);
	/*std::cout << "Vector normal DOUBLE: " << endl;
	for (int i = 0; i < cell; i++){
		std::cout << "(" << vec_norm_ecef.x[i] << ", " << vec_norm_ecef.y[i] << ", " << vec_norm_ecef.z[i] << ")" << " ";
	}
	std::cout << endl;*/

	vec_north_ecef = north_dir(circ_ecef.x, circ_ecef.y, circ_ecef.z, cell, vec_norm_ecef);
	/*std::cout << " Vector that points towrds North DOUBLE: " << endl;
	for (int i = 0; i < cell; i++){
		std::cout << "(" << vec_north_ecef.x[i] << ", " << vec_north_ecef.y[i] << ", " << vec_north_ecef.z[i] << ")" << " ";
	}
	std::cout << endl << endl << endl;*/

	coords vec_norm_enu, vec_north_enu;
	vec_norm_enu = ecef2enu(vec_norm_ecef.x, vec_norm_ecef.y, vec_norm_ecef.z, cell, lon_or, lat_or);
	vec_north_enu = ecef2enu(vec_north_ecef.x, vec_north_ecef.y, vec_north_ecef.z, cell, lon_or, lat_or);

	// Transfor the Norm and North vectors from double to float

	vector<float> vec_norm_ecef_x(vec_norm_ecef.x.begin(), vec_norm_ecef.x.end());
	vector<float> vec_norm_ecef_y(vec_norm_ecef.y.begin(), vec_norm_ecef.y.end());
	vector<float> vec_norm_ecef_z(vec_norm_ecef.z.begin(), vec_norm_ecef.z.end());

	vector<float> vec_north_ecef_x(vec_north_ecef.x.begin(), vec_north_ecef.x.end());
	vector<float> vec_north_ecef_y(vec_north_ecef.y.begin(), vec_north_ecef.y.end());
	vector<float> vec_north_ecef_z(vec_north_ecef.z.begin(), vec_north_ecef.z.end());

	vector<float> circ_ecef_x(circ_ecef.x.begin(), circ_ecef.x.end());
	vector<float> circ_ecef_y(circ_ecef.y.begin(), circ_ecef.y.end());
	vector<float> circ_ecef_z(circ_ecef.z.begin(), circ_ecef.z.end());

	/*
	// ENU CASE
	vector<float> vec_norm_enu_x(vec_norm_enu.x.begin(), vec_norm_enu.x.end());
	vector<float> vec_norm_enu_y(vec_norm_enu.y.begin(), vec_norm_enu.y.end());
	vector<float> vec_norm_enu_z(vec_norm_enu.z.begin(), vec_norm_enu.z.end());

	vector<float> vec_north_enu_x(vec_north_enu.x.begin(), vec_north_enu.x.end());
	vector<float> vec_north_enu_y(vec_north_enu.y.begin(), vec_north_enu.y.end());
	vector<float> vec_north_enu_z(vec_north_enu.z.begin(), vec_north_enu.z.end());
	*/

    // ------------------------------------------------------------------------
  	// Compute and save horizon in one iteration
    // ------------------------------------------------------------------------

    if (hori_buffer_size <= hori_buffer_size_max) {
    
    	cout << "Compute and save horizon in one iteration" << endl;

    	// --------------------------------------------------------------------
  		// Perform ray tracing
    	// --------------------------------------------------------------------

  		auto start_ray = std::chrono::high_resolution_clock::now();
    
		for (int i = 0; i < 1; i++){
			std::cout << "Cell nr."<< i << endl;
			if (mask[i] == 1){

  					// Ray origin ECEF CASE
					
  					float ray_org_x = circ_ecef_x[i] 
  						+ vec_norm_ecef_x[i] * ray_org_elev;
  					float ray_org_y = circ_ecef_y[i] 
  						+ vec_norm_ecef_y[i] * ray_org_elev;
  					float ray_org_z = circ_ecef_z[i] 
  						+ vec_norm_ecef_z[i] * ray_org_elev;

					std::cout << "Ray origin: (" << ray_org_x << ", " << ray_org_y << ", " << ray_org_z << ")" << endl;
					
					std::cout << "Vec norm ecef: (" << vec_norm_ecef_x[i] << ", " << vec_norm_ecef_y[i] << ", " << vec_norm_ecef_z[i] << ")" << endl;
					std::cout << "Vec north ecef: (" << vec_north_ecef_x[i] << ", " << vec_north_ecef_y[i] << ", " << vec_north_ecef_z[i] << ")" << endl;
					
					ray_guess_const(ray_org_x, ray_org_y, ray_org_z, azim_num, 
						hori_acc, dist_search, elev_ang_low_lim, elev_ang_up_lim, 
						scene, num_rays, hori_buffer, vec_norm_ecef_x[i], 
						vec_norm_ecef_y[i], vec_norm_ecef_z[i], vec_north_ecef_x[i], 
						vec_north_ecef_y[i], vec_north_ecef_z[i], azim_sin, azim_cos);
					
					
					/*
					//Ray origin ENU CASE
  					float ray_org_x = circ_enu.x[i] 
  						+ vec_norm_enu_x[i] * ray_org_elev;
  					float ray_org_y = circ_enu.y[i] 
  						+ vec_norm_enu_y[i] * ray_org_elev;
  					float ray_org_z = circ_enu.z[i] 
  						+ vec_norm_enu_z[i] * ray_org_elev;

					std::cout << "Ray origin: (" << ray_org_x << ", " << ray_org_y << ", " << ray_org_z << ")" << endl;
					
					std::cout << "Vec norm ecef: (" << vec_norm_enu_x[i] << ", " << vec_norm_enu_y[i] << ", " << vec_norm_enu_z[i] << ")" << endl;
					std::cout << "Vec north ecef: (" << vec_north_enu_x[i] << ", " << vec_north_enu_y[i] << ", " << vec_north_enu_z[i] << ")" << endl;
					
					ray_guess_const(ray_org_x, ray_org_y, ray_org_z, azim_num, 
						hori_acc, dist_search, elev_ang_low_lim, elev_ang_up_lim, 
						scene, num_rays, hori_buffer, vec_norm_enu_x[i], 
						vec_norm_enu_y[i], vec_norm_enu_z[i], vec_north_enu_x[i], 
						vec_north_enu_y[i], vec_north_enu_z[i], azim_sin, azim_cos);
					*/

					std::cout << endl;

			}else{

			}
		}

    
  		auto end_ray = std::chrono::high_resolution_clock::now();
  		time_ray += (end_ray - start_ray);
	}

    return ;
}