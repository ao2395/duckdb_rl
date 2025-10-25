//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_model_interface.cpp
//
//
//===----------------------------------------------------------------------===//

#include "duckdb/main/rl_model_interface.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/printer.hpp"
#include "duckdb/optimizer/rl_feature_collector.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_any_join.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"

namespace duckdb {

RLModelInterface::RLModelInterface(ClientContext &context) : context(context), enabled(true) {
	// Always enabled for now
}

string OperatorFeatures::ToString() const {
	string result = "\n[RL MODEL] ========== OPERATOR FEATURES ==========\n";
	result += "[RL MODEL] Operator Type: " + operator_type + "\n";
	result += "[RL MODEL] Operator Name: " + operator_name + "\n";
	result += "[RL MODEL] DuckDB Estimated Cardinality: " + std::to_string(estimated_cardinality) + "\n";

	// TABLE SCAN STATS (matching original format)
	if (base_table_cardinality > 0) {
		result += "[RL MODEL] ===== TABLE SCAN STATS =====\n";
		if (!table_name.empty()) {
			result += "[RL MODEL] Table Name: " + table_name + "\n";
		}
		result += "[RL MODEL] Base Table Cardinality: " + std::to_string(base_table_cardinality) + "\n";

		if (!column_distinct_counts.empty()) {
			for (auto &entry : column_distinct_counts) {
				result += "[RL MODEL] Column: " + entry.first + " | Distinct Count (HLL): " + std::to_string(entry.second) + "\n";
			}
		}

		if (num_table_filters > 0) {
			result += "[RL MODEL] Number of table filters: " + std::to_string(num_table_filters) + "\n";

			// Filter inspection details with child count tracking
			idx_t child_count = 0;
			for (idx_t i = 0; i < filter_types.size(); i++) {
				if (i < filter_column_ids.size() && child_count == 0) {
					result += "[RL MODEL] --- Filter Inspection on column " + std::to_string(filter_column_ids[i]) + " ---\n";
				}
				result += "[RL MODEL] Filter Type: " + filter_types[i] + "\n";

				// Track CONJUNCTION_AND to count children
				if (filter_types[i] == "CONJUNCTION_AND") {
					// Count upcoming CONSTANT_COMPARISON children
					idx_t num_children = 0;
					for (idx_t j = i + 1; j < filter_types.size() && filter_types[j] != "CONJUNCTION_AND"; j++) {
						if (filter_types[j] == "CONSTANT_COMPARISON") {
							num_children++;
						}
					}
					if (num_children > 0) {
						result += "[RL MODEL] Number of AND child filters: " + std::to_string(num_children) + "\n";
						child_count = num_children;
					}
				} else if (child_count > 0) {
					child_count--;
					result += "[RL MODEL] --- Filter Inspection on column " + std::to_string(filter_column_ids[0]) + " ---\n";
				}

				if (i < comparison_types.size() && !comparison_types[i].empty()) {
					result += "[RL MODEL] Comparison Type: " + comparison_types[i] + "\n";
					if (comparison_types[i] != "EQUAL") {
						result += "[RL MODEL] Non-equality comparison - no selectivity applied\n";
					}
				}
			}

			if (used_default_selectivity) {
				result += "[RL MODEL] Using DEFAULT_SELECTIVITY: 0.200000\n";
				result += "[RL MODEL] Cardinality after default selectivity: " + std::to_string(cardinality_after_default_selectivity) + "\n";
			}
		}

		if (final_cardinality > 0) {
			result += "[RL MODEL] Final Cardinality (after filters): " + std::to_string(final_cardinality) + "\n";
			result += "[RL MODEL] Filter Selectivity Ratio: " + std::to_string(filter_selectivity) + "\n";
		}
		result += "[RL MODEL] ===== END TABLE SCAN STATS =====\n";
	}

	// JOIN FEATURES (matching original format)
	if (!join_type.empty()) {
		result += "[RL MODEL] ===== CARDINALITY ESTIMATION START =====\n";
		if (!join_relation_set.empty()) {
			result += "[RL MODEL] Join Relation Set: " + join_relation_set + "\n";
			result += "[RL MODEL] Number of relations in join: " + std::to_string(num_relations) + "\n";
		}
		result += "[RL MODEL] Join Type: " + join_type + "\n";
		if (left_relation_card > 0 && right_relation_card > 0) {
			result += "[RL MODEL] Left Relation Cardinality: " + std::to_string(left_relation_card) + "\n";
			result += "[RL MODEL] Right Relation Cardinality: " + std::to_string(right_relation_card) + "\n";
			result += "[RL MODEL] Left Denominator: " + std::to_string(left_denominator) + "\n";
			result += "[RL MODEL] Right Denominator: " + std::to_string(right_denominator) + "\n";
		} else {
			result += "[RL MODEL] Left Cardinality: " + std::to_string(left_cardinality) + "\n";
			result += "[RL MODEL] Right Cardinality: " + std::to_string(right_cardinality) + "\n";
		}
		if (!comparison_type_join.empty()) {
			result += "[RL MODEL] Comparison Type: " + comparison_type_join + "\n";
		}
		if (tdom_from_hll) {
			result += "[RL MODEL] TDOM from HLL: true\n";
		}
		if (tdom_value > 0) {
			result += "[RL MODEL] TDOM value: " + std::to_string(tdom_value) + "\n";
			if (extra_ratio > 1.0) {
				result += "[RL MODEL] Equality Join - Extra Ratio: " + std::to_string(extra_ratio) + "\n";
			}
		}
		if (numerator > 0 && denominator > 0) {
			result += "[RL MODEL] Numerator (product of cardinalities): " + std::to_string(numerator) + "\n";
			result += "[RL MODEL] Denominator (TDOM-based): " + std::to_string(denominator) + "\n";
			double calc_estimate = numerator / denominator;
			result += "[RL MODEL] Estimated Cardinality: " + std::to_string(calc_estimate) + "\n";
		}
		result += "[RL MODEL] ===== CARDINALITY ESTIMATION END =====\n";
	}

	// AGGREGATE STATS (matching original format)
	if (num_group_by_columns > 0 || num_aggregate_functions > 0) {
		result += "[RL MODEL] ===== AGGREGATE STATISTICS =====\n";
		result += "[RL MODEL] Number of GROUP BY columns: " + std::to_string(num_group_by_columns) + "\n";
		result += "[RL MODEL] Number of aggregate functions: " + std::to_string(num_aggregate_functions) + "\n";
		result += "[RL MODEL] Number of grouping sets: " + std::to_string(num_grouping_sets) + "\n";
		result += "[RL MODEL] ===== END AGGREGATE STATISTICS =====\n";
	}

	// FILTER FEATURES (for standalone filters)
	if (!filter_types.empty() && base_table_cardinality == 0) {
		result += "[RL MODEL] Filter Types: ";
		for (idx_t i = 0; i < filter_types.size(); i++) {
			result += filter_types[i];
			if (i < filter_types.size() - 1) result += ", ";
		}
		result += "\n";

		if (!comparison_types.empty()) {
			result += "[RL MODEL] Comparison Types: ";
			for (idx_t i = 0; i < comparison_types.size(); i++) {
				result += comparison_types[i];
				if (i < comparison_types.size() - 1) result += ", ";
			}
			result += "\n";
		}
	}

	result += "[RL MODEL] ============================================\n";
	return result;
}

OperatorFeatures RLModelInterface::ExtractFeatures(LogicalOperator &op, ClientContext &context) {
	OperatorFeatures features;

	// Basic operator info
	features.operator_type = LogicalOperatorToString(op.type);
	features.operator_name = op.GetName();
	features.estimated_cardinality = op.estimated_cardinality;

	// Try to get features from the collector (populated during statistics propagation)
	auto &collector = RLFeatureCollector::Get();

	// Extract operator-specific features
	switch (op.type) {
	case LogicalOperatorType::LOGICAL_GET: {
		auto &get = op.Cast<LogicalGet>();
		if (get.function.cardinality) {
			auto card_stats = get.function.cardinality(context, get.bind_data.get());
			if (card_stats) {
				features.base_table_cardinality = card_stats->estimated_cardinality;
			}
		}

		// Get detailed table scan features from collector
		auto table_features = collector.GetTableScanFeatures(&op);
		if (table_features) {
			features.table_name = table_features->table_name;
			features.base_table_cardinality = table_features->base_cardinality;
			features.column_distinct_counts = table_features->column_distinct_counts;
			features.num_table_filters = table_features->num_table_filters;
			features.final_cardinality = table_features->final_cardinality;
			features.filter_selectivity = table_features->filter_selectivity;
			features.used_default_selectivity = table_features->used_default_selectivity;
			features.cardinality_after_default_selectivity = table_features->cardinality_after_default_selectivity;
			features.filter_types = table_features->filter_types;
			features.comparison_types = table_features->comparison_types;
			features.filter_column_ids = table_features->filter_column_ids;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_FILTER: {
		auto &filter = op.Cast<LogicalFilter>();
		// Extract filter expression types
		for (auto &expr : filter.expressions) {
			features.filter_types.push_back(ExpressionTypeToString(expr->type));
		}

		// Get detailed filter features from collector
		auto filter_features = collector.GetFilterFeatures(&op);
		if (filter_features) {
			features.comparison_types = filter_features->comparison_types;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_COMPARISON_JOIN: {
		auto &join = op.Cast<LogicalComparisonJoin>();
		features.join_type = JoinTypeToString(join.join_type);
		if (op.children.size() >= 2) {
			features.left_cardinality = op.children[0]->estimated_cardinality;
			features.right_cardinality = op.children[1]->estimated_cardinality;
		}

		// Try to get detailed join features from collector (by operator or by estimated cardinality)
		auto join_features = collector.GetJoinFeatures(&op);
		if (!join_features && op.estimated_cardinality > 0) {
			// Try matching by estimated cardinality
			join_features = collector.GetJoinFeaturesByEstimate(op.estimated_cardinality);
		}
		if (join_features) {
			features.tdom_value = join_features->tdom_value;
			features.tdom_from_hll = join_features->tdom_from_hll;
			features.join_relation_set = join_features->join_relation_set;
			features.num_relations = join_features->num_relations;
			features.left_relation_card = join_features->left_relation_card;
			features.right_relation_card = join_features->right_relation_card;
			features.left_denominator = join_features->left_denominator;
			features.right_denominator = join_features->right_denominator;
			features.comparison_type_join = join_features->comparison_type;
			features.extra_ratio = join_features->extra_ratio;
			features.numerator = join_features->numerator;
			features.denominator = join_features->denominator;
		}
		break;
	}
	case LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY: {
		auto &aggr = op.Cast<LogicalAggregate>();
		features.num_group_by_columns = aggr.groups.size();
		features.num_aggregate_functions = aggr.expressions.size();
		features.num_grouping_sets = aggr.grouping_sets.size();
		break;
	}
	default:
		// For other operators, just use basic info
		break;
	}

	return features;
}

idx_t RLModelInterface::GetCardinalityEstimate(const OperatorFeatures &features) {
	if (!enabled) {
		return 0; // Don't override
	}

	// Print all features received by the model
	Printer::Print(features.ToString());

	// For now, pass through DuckDB's estimate
	// Later this will be replaced with actual RL model inference
	Printer::Print("[RL MODEL] Returning DuckDB estimate: " + std::to_string(features.estimated_cardinality) + "\n");
	return features.estimated_cardinality;
}

void RLModelInterface::TrainModel(const OperatorFeatures &features, idx_t actual_cardinality) {
	// To be implemented later for training
	// This will be called after each operator executes with the actual cardinality
}

} // namespace duckdb
