#include "traversal.hpp"
#include "rays.hpp"

namespace gw {
	SampleResult TraversalEngine::traverse_batch(const std::vector<Ray>& rays) {
		SampleResult r{};
		// For now, produce one scalar feature per ray as a placeholder
		r.features.resize(rays.size(), 0.5f);
		return r;
	}
}