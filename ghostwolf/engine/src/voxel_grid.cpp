#include "voxel_grid.hpp"

namespace gw {
	bool VoxelGrid::load_from_model(const std::string&, const std::string&) { return true; }
	bool VoxelGrid::build_from_tensors(const VoxelGridConfig& cfg) {
		config_ = cfg;
		return true;
	}
}