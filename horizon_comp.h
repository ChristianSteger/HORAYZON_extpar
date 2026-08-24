#include <embree4/rtcore.h>

namespace shapes {

struct geom_vector{
    double x, y, z;

    geom_vector operator-(const geom_vector& other) const {
        return {x - other.x,
                y - other.y,
                z - other.z};
    }

    geom_vector operator*(double s) const {
        return {s * x, s * y, s * z};
    }

};

struct geom_point{
    double x, y, z;

    geom_vector operator-(const geom_point& other) const {
        return {x - other.x,
                y - other.y,
                z - other.z};
    }

    geom_point operator+(const geom_vector& other) const {
        return {x + other.x,
                y + other.y,
                z + other.z};
    }

};

class CppTerrain {
public:
    RTCDevice device = nullptr;
    RTCScene scene = nullptr;
    float* tri_vert_cl;
    unsigned int* tri_face_cl;
    int num_vert_cl;
    int num_face_cl;
    geom_point earth_centre_cl;
    geom_point north_pole_cl;
    CppTerrain();
    ~CppTerrain();
    void initialise(
        float* tri_vert,
        unsigned int* tri_face,
        int num_vert,
        int num_face,
        float* earth_centre,
        float* north_pole
    );
    void horizon_vertex(
        int num_azim,
        double azim_offset,
        unsigned int* slice_loc,
        float dist_search,
        double ray_origin_elev,
        double horizon_acc,
        double elev_angle_min,
        float* horizon
    );
    void horizon_centroid(
        int num_azim,
        unsigned int* slice_loc,
        float dist_search,
        double ray_origin_elev,
        double horizon_acc,
        double elev_angle_min,
        float* horizon
    );
};
}