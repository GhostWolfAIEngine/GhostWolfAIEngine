#pragma once

#include <vector>

namespace gw {
	class Aggregator {
	public:
		// Collapse 120 rays to mean logit vector, return selected token id and logprob
		int infer_token(const std::vector<float>& per_ray_features, float& out_logprob);
	};
}