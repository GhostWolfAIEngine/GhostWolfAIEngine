#include <iostream>
#include <cstdint>
#include "gw_engine.hpp"

int main() {
	GW_Handle h = gw_create();
	if (!h) { std::cerr << "Failed to create engine" << std::endl; return 1; }
	gw_load_model(h, "demo.onnx", GW_ONNX);
	gw_voxelize(h, "voxel_config = true");
	gw_infer_begin(h, "{\"prompt\":\"Hello GhostWolf\"}");
	for (int i = 0; i < 5; ++i) {
		int32_t token = -1; float lp = 0.0f;
		gw_infer_step(h, &token, &lp);
		std::cout << "Step " << i << ": token=" << token << ", lp=" << lp << std::endl;
	}
	gw_destroy(h);
	return 0;
}