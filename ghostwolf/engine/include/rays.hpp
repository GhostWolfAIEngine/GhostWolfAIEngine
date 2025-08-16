#pragma once

#include <cstdint>
#include <vector>

namespace gw {
	struct Ray {
		float origin[4]; // x,y,z,t
		float direction[4];
		float max_distance;
		uint32_t group_id; // 0=past,1=present,2=future
	};

	class RayGenerator {
	public:
		std::vector<Ray> generate_4d_rays(uint32_t total_rays = 120) const;
	};
}