#include "aggregation.hpp"
#include <numeric>

namespace gw {
	int Aggregator::infer_token(const std::vector<float>& per_ray_features, float& out_logprob) {
		if (per_ray_features.empty()) { out_logprob = 0.0f; return 0; }
		float mean = std::accumulate(per_ray_features.begin(), per_ray_features.end(), 0.0f) / per_ray_features.size();
		out_logprob = mean; // placeholder
		int token_id = static_cast<int>(mean * 100.0f) % 100; // fake mapping
		if (token_id < 0) token_id = -token_id;
		return token_id;
	}
}