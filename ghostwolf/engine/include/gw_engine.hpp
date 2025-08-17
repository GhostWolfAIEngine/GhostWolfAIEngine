#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

extern "C" {
	typedef void* GW_Handle;
	enum GW_ModelFormat : uint32_t { GW_GGUF = 0, GW_ONNX = 1, GW_SAFETENSORS = 2 };

	GW_Handle gw_create();
	int32_t gw_load_model(GW_Handle h, const char* path, GW_ModelFormat fmt);
	int32_t gw_voxelize(GW_Handle h, const char* config_toml);
	int32_t gw_infer_begin(GW_Handle h, const char* prompt_json);
	int32_t gw_infer_step(GW_Handle h, /*out*/ int32_t* token_id, /*out*/ float* logprob);
	void gw_cancel(GW_Handle h);
	void gw_destroy(GW_Handle h);
}

namespace gw {
	struct InferenceSettings {
		int32_t max_tokens { 64 };
		int32_t ray_splits { 120 };
		float temperature { 0.8f };
	};

	class EngineImpl; // forward declaration

	class Engine {
	public:
		Engine();
		~Engine();

		int32_t load_model(const std::string& path, GW_ModelFormat fmt);
		int32_t voxelize(const std::string& config_toml);
		int32_t infer_begin(const std::string& prompt_json);
		int32_t infer_step(int32_t& out_token_id, float& out_logprob);
		void cancel();

	private:
		std::unique_ptr<EngineImpl> pimpl_;
	};
}