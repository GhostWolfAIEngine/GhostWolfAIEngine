#pragma once

#include <cstdint>

namespace gw {
	class Scheduler {
	public:
		void prefetch_async();
		void submit_traversal();
		void wait_for_completion();
	};
}