#include "traversal.hpp"
#include "rays.hpp"

namespace gw {
	SampleResult TraversalEngine::traverse_batch(const std::vector<Ray>& rays) {
		SampleResult r{};
		r.features.resize(rays.size());
		for (size_t i = 0; i < rays.size(); ++i) {
			// Simple diagnostic: encode group id into feature value
			// past=0 -> 0.3, present=1 -> 0.6, future=2 -> 0.9
			switch (rays[i].group_id) {
				case 0: r.features[i] = 0.3f; break;
				case 1: r.features[i] = 0.6f; break;
				default: r.features[i] = 0.9f; break;
			}
		}
		return r;
	}
}