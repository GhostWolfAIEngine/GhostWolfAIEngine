#include "rays.hpp"
#include <cmath>

namespace gw {
	static void normalize4(float v[4]) {
		float n = std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+v[3]*v[3]);
		if (n > 0.0f) { v[0]/=n; v[1]/=n; v[2]/=n; v[3]/=n; }
	}

	std::vector<Ray> RayGenerator::generate_4d_rays(uint32_t total_rays) const {
		if (total_rays < 3) total_rays = 3;
		uint32_t per_group = total_rays / 3;
		std::vector<Ray> rays;
		rays.reserve(per_group * 3);
		for (uint32_t g = 0; g < 3; ++g) {
			for (uint32_t i = 0; i < per_group; ++i) {
				float u = (i + 0.5f) / per_group;
				float theta = 6.2831853f * u; // 2pi
				float phi = 3.1415926f * ((i % per_group) + 0.5f) / per_group;
				Ray r{};
				r.origin[0] = 0.0f; r.origin[1] = 0.0f; r.origin[2] = 0.0f; r.origin[3] = (g==0?-1.0f:(g==1?0.0f:1.0f));
				r.direction[0] = std::sin(phi) * std::cos(theta);
				r.direction[1] = std::sin(phi) * std::sin(theta);
				r.direction[2] = std::cos(phi);
				r.direction[3] = (g==0?-1.0f:(g==1?0.0f:1.0f));
				r.max_distance = 1.0f;
				r.group_id = g;
				normalize4(r.direction);
				rays.push_back(r);
			}
		}
		return rays;
	}
}