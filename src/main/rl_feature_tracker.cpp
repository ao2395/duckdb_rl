//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_feature_tracker.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/main/rl_feature_tracker.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/common/printer.hpp"
#include <mutex>

namespace duckdb {

RLFeatureTracker::RLFeatureTracker(ClientContext &context) : context(context), enabled(true) {
	// RL feature tracking is always enabled
}

void RLFeatureTracker::StartOperator(optional_ptr<const PhysicalOperator> phys_op) {
	if (!enabled || !phys_op) {
		return;
	}

	std::lock_guard<std::mutex> guard(lock);
	auto &stats = operator_stats[*phys_op];

	// Only initialize once per operator (multiple workers may call this)
	if (stats.estimated_cardinality == 0) {
		stats.operator_name = phys_op->GetName();
		stats.estimated_cardinality = phys_op->estimated_cardinality;
	}
}

void RLFeatureTracker::EndOperator(optional_ptr<const PhysicalOperator> phys_op, idx_t actual_rows) {
	if (!enabled || !phys_op || actual_rows == 0) {
		return;
	}

	// No lock needed - atomic add handles thread safety
	auto it = operator_stats.find(*phys_op);
	if (it != operator_stats.end()) {
		it->second.AddActualRows(actual_rows);
	}
}

void RLFeatureTracker::Finalize() {
	if (!enabled) {
		return;
	}

	std::lock_guard<std::mutex> guard(lock);

	for (auto &entry : operator_stats) {
		auto &stats = entry.second;
		idx_t actual_count = stats.actual_cardinality.load();

		if (actual_count > 0) {
			// Printer::Print("\n[RL FEATURE] *** ACTUAL CARDINALITY *** Operator: " + stats.operator_name +
			//                " | Actual Output: " + std::to_string(actual_count) +
			//                " | Estimated: " + std::to_string(stats.estimated_cardinality));

			if (stats.estimated_cardinality > 0) {
				double error = static_cast<double>(actual_count) / static_cast<double>(stats.estimated_cardinality);
				if (error < 1.0) {
					error = 1.0 / error;
				}
				// Printer::Print("[RL FEATURE] *** Q-ERROR *** " + std::to_string(error));
			}
		}
	}
}

void RLFeatureTracker::Reset() {
	if (!enabled) {
		return;
	}

	std::lock_guard<std::mutex> guard(lock);
	operator_stats.clear();
}

} // namespace duckdb
