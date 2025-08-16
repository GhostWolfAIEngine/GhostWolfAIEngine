#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <cstdlib>
#include "gw_engine.hpp"

static GW_ModelFormat parse_fmt(const std::string& s) {
	if (s == "gguf") return GW_GGUF;
	if (s == "onnx") return GW_ONNX;
	if (s == "safetensors") return GW_SAFETENSORS;
	return GW_ONNX;
}

int main(int argc, char** argv) {
	std::string model_path = "demo.onnx";
	std::string fmt_str = "onnx";
	std::string prompt = "Hello GhostWolf";
	int steps = 5;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--model" && i + 1 < argc) { model_path = argv[++i]; }
		else if (arg == "--fmt" && i + 1 < argc) { fmt_str = argv[++i]; }
		else if (arg == "--prompt" && i + 1 < argc) { prompt = argv[++i]; }
		else if (arg == "--steps" && i + 1 < argc) { steps = std::atoi(argv[++i]); }
		else if (arg == "-h" || arg == "--help") {
			std::cout << "Usage: ghostwolf_cli --model <path> --fmt <gguf|onnx|safetensors> --prompt <text> --steps <N>" << std::endl;
			return 0;
		}
	}

	GW_Handle h = gw_create();
	if (!h) { std::cerr << "Failed to create engine" << std::endl; return 1; }

	std::cout << "Loading model: " << model_path << " (fmt=" << fmt_str << ")" << std::endl;
	gw_load_model(h, model_path.c_str(), parse_fmt(fmt_str));

	gw_voxelize(h, "voxel_config = true");

	std::string prompt_json = std::string("{\"prompt\":\"") + prompt + "\"}";
	std::cout << "Prompt: " << prompt << std::endl;
	gw_infer_begin(h, prompt_json.c_str());

	for (int i = 0; i < steps; ++i) {
		int32_t token = -1; float lp = 0.0f;
		gw_infer_step(h, &token, &lp);
		std::cout << "Step " << i << ": token=" << token << ", lp=" << lp << std::endl;
	}
	gw_destroy(h);
	return 0;
}