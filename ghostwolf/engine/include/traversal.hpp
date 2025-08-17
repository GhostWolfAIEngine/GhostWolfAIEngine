#pragma once

#include <cstdint>
#include <vector>
#include "rays.hpp"

namespace gw {
	struct SampleResult {
		std::vector<float> features; // integrated features per ray
	};

	class TraversalEngine {
	public:
		SampleResult traverse_batch(const std::vector<Ray>& rays);
	};
}