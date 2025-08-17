#include "gw_engine.hpp"
#include "rays.hpp"
#include "traversal.hpp"
#include "aggregation.hpp"

#include <memory>

namespace gw {
	class EngineImpl {
	public:
		int32_t load_model(const std::string&, GW_ModelFormat) { return 0; }
		int32_t voxelize(const std::string&) { return 0; }
		int32_t infer_begin(const std::string&) { return 0; }
		int32_t infer_step(int32_t& token_id, float& logprob) {
			RayGenerator gen;
			auto rays = gen.generate_4d_rays(120);
			TraversalEngine trav;
			auto samples = trav.traverse_batch(rays);
			Aggregator agg;
			token_id = agg.infer_token(samples.features, logprob);
			return 0;
		}
		void cancel() {}
	};

	Engine::Engine() : pimpl_(std::make_unique<EngineImpl>()) {}
	Engine::~Engine() = default;

	int32_t Engine::load_model(const std::string& path, GW_ModelFormat fmt) { return pimpl_->load_model(path, fmt); }
	int32_t Engine::voxelize(const std::string& config_toml) { return pimpl_->voxelize(config_toml); }
	int32_t Engine::infer_begin(const std::string& prompt_json) { return pimpl_->infer_begin(prompt_json); }
	int32_t Engine::infer_step(int32_t& out_token_id, float& out_logprob) { return pimpl_->infer_step(out_token_id, out_logprob); }
	void Engine::cancel() { pimpl_->cancel(); }

}

using gw::Engine;

struct GW_Wrap { Engine engine; };

extern "C" {
	GW_Handle gw_create() { return new GW_Wrap(); }
	int32_t gw_load_model(GW_Handle h, const char* path, GW_ModelFormat fmt) {
		return reinterpret_cast<GW_Wrap*>(h)->engine.load_model(path ? path : std::string(), fmt);
	}
	int32_t gw_voxelize(GW_Handle h, const char* config_toml) {
		return reinterpret_cast<GW_Wrap*>(h)->engine.voxelize(config_toml ? config_toml : std::string());
	}
	int32_t gw_infer_begin(GW_Handle h, const char* prompt_json) {
		return reinterpret_cast<GW_Wrap*>(h)->engine.infer_begin(prompt_json ? prompt_json : std::string());
	}
	int32_t gw_infer_step(GW_Handle h, int32_t* token_id, float* logprob) {
		int32_t tid = -1; float lp = 0.0f;
		int32_t rc = reinterpret_cast<GW_Wrap*>(h)->engine.infer_step(tid, lp);
		if (token_id) *token_id = tid;
		if (logprob) *logprob = lp;
		return rc;
	}
	void gw_cancel(GW_Handle h) {
		reinterpret_cast<GW_Wrap*>(h)->engine.cancel();
	}
	void gw_destroy(GW_Handle h) {
		delete reinterpret_cast<GW_Wrap*>(h);
	}
}