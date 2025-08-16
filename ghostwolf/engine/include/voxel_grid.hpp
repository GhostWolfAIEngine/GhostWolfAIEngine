#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace gw {
	struct VoxelBrickHeader {
		uint32_t brick_dim; // e.g., 32
		uint32_t t_slices;  // time dimension slices
		float scale;
		float zero;
	};

	struct VoxelBrick {
		VoxelBrickHeader header {};
		std::vector<uint8_t> compressed; // INT8 payload per feature channel
	};

	struct VoxelGridIndexEntry {
		int32_t x, y, z, t;
		uint64_t brick_offset;
	};

	struct VoxelGridConfig {
		uint32_t brick_dim { 32 };
		uint32_t t_slices { 9 }; // past..future slices
		uint32_t feature_channels { 64 };
	};

	class VoxelGrid {
	public:
		VoxelGrid() = default;
		~VoxelGrid() = default;

		bool load_from_model(const std::string& model_path, const std::string& fmt);
		bool build_from_tensors(const VoxelGridConfig& cfg);
		const VoxelGridConfig& config() const { return config_; }

	private:
		VoxelGridConfig config_ {};
		std::vector<VoxelGridIndexEntry> index_;
		std::vector<VoxelBrick> bricks_;
	};
}