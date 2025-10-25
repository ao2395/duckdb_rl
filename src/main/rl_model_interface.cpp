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

vector<double> RLModelInterface::FeaturesToVector(const OperatorFeatures &features) {
	vector<double> feature_vec(FEATURE_VECTOR_SIZE, 0.0);
	idx_t idx = 0;

	// Helper lambda for safe log (avoid log(0))
	auto safe_log = [](idx_t val) -> double {
		return val > 0 ? std::log(static_cast<double>(val)) : 0.0;
	};

	// 1. OPERATOR TYPE (One-hot encoding) - 10 features
	// GET, JOIN, FILTER, AGGREGATE, PROJECTION, TOP_N, ORDER_BY, LIMIT, UNION, OTHER
	if (!features.table_name.empty()) {
		feature_vec[idx] = 1.0; // GET
	} else if (!features.join_type.empty()) {
		feature_vec[idx + 1] = 1.0; // JOIN
	} else if (!features.filter_types.empty() && features.table_name.empty()) {
		feature_vec[idx + 2] = 1.0; // FILTER
	} else if (features.num_group_by_columns > 0 || features.num_aggregate_functions > 0) {
		feature_vec[idx + 3] = 1.0; // AGGREGATE
	} else {
		feature_vec[idx + 9] = 1.0; // OTHER (PROJECTION, TOP_N, etc.)
	}
	idx += 10;

	// 2. TABLE SCAN FEATURES - 8 features
	if (!features.table_name.empty()) {
		feature_vec[idx++] = safe_log(features.base_table_cardinality);
		feature_vec[idx++] = static_cast<double>(features.num_table_filters);
		feature_vec[idx++] = features.filter_selectivity;
		feature_vec[idx++] = features.used_default_selectivity ? 1.0 : 0.0;
		feature_vec[idx++] = static_cast<double>(features.filter_types.size());

		// Column distinct count statistics
		if (!features.column_distinct_counts.empty() && features.base_table_cardinality > 0) {
			double sum = 0.0, min_ratio = 1.0, max_ratio = 0.0;
			for (const auto &entry : features.column_distinct_counts) {
				double ratio = static_cast<double>(entry.second) / static_cast<double>(features.base_table_cardinality);
				sum += ratio;
				min_ratio = std::min(min_ratio, ratio);
				max_ratio = std::max(max_ratio, ratio);
			}
			feature_vec[idx++] = sum / features.column_distinct_counts.size(); // avg ratio
			feature_vec[idx++] = max_ratio;
			feature_vec[idx++] = min_ratio;
		} else {
			idx += 3;
		}
	} else {
		idx += 8;
	}

	// 3. JOIN FEATURES - 21 features
	if (!features.join_type.empty()) {
		feature_vec[idx++] = safe_log(features.left_cardinality);
		feature_vec[idx++] = safe_log(features.right_cardinality);
		feature_vec[idx++] = safe_log(features.tdom_value);
		feature_vec[idx++] = features.tdom_from_hll ? 1.0 : 0.0;

		// Join type one-hot (INNER, LEFT, RIGHT, SEMI, ANTI)
		if (features.join_type == "INNER") feature_vec[idx] = 1.0;
		else if (features.join_type == "LEFT") feature_vec[idx + 1] = 1.0;
		else if (features.join_type == "RIGHT") feature_vec[idx + 2] = 1.0;
		else if (features.join_type == "SEMI") feature_vec[idx + 3] = 1.0;
		else if (features.join_type == "ANTI") feature_vec[idx + 4] = 1.0;
		idx += 5;

		// Comparison type one-hot (EQUAL, LT, GT, LTE, GTE, NEQ)
		if (features.comparison_type_join == "EQUAL") feature_vec[idx] = 1.0;
		else if (features.comparison_type_join == "LESSTHAN") feature_vec[idx + 1] = 1.0;
		else if (features.comparison_type_join == "GREATERTHAN") feature_vec[idx + 2] = 1.0;
		else if (features.comparison_type_join == "LESSTHANOREQUALTO") feature_vec[idx + 3] = 1.0;
		else if (features.comparison_type_join == "GREATERTHANOREQUALTO") feature_vec[idx + 4] = 1.0;
		else if (features.comparison_type_join == "NOTEQUAL") feature_vec[idx + 5] = 1.0;
		idx += 6;

		feature_vec[idx++] = safe_log(static_cast<idx_t>(features.extra_ratio));
		feature_vec[idx++] = std::log(std::max(1.0, features.numerator));
		feature_vec[idx++] = std::log(std::max(1.0, features.denominator));
		feature_vec[idx++] = static_cast<double>(features.num_relations);
		feature_vec[idx++] = std::log(std::max(1.0, features.left_denominator));
		feature_vec[idx++] = std::log(std::max(1.0, features.right_denominator));
	} else {
		idx += 21;
	}

	// 4. AGGREGATE FEATURES - 4 features
	if (features.num_group_by_columns > 0 || features.num_aggregate_functions > 0) {
		feature_vec[idx++] = safe_log(features.estimated_cardinality); // Input from child
		feature_vec[idx++] = static_cast<double>(features.num_group_by_columns);
		feature_vec[idx++] = static_cast<double>(features.num_aggregate_functions);
		feature_vec[idx++] = static_cast<double>(features.num_grouping_sets);
	} else {
		idx += 4;
	}

	// 5. FILTER FEATURES - 2 features
	if (!features.filter_types.empty() && features.table_name.empty()) {
		feature_vec[idx++] = safe_log(features.estimated_cardinality); // Input from child
		feature_vec[idx++] = static_cast<double>(features.filter_types.size());
	} else {
		idx += 2;
	}

	// 6. CONTEXT FEATURES - 1 feature
	feature_vec[idx++] = safe_log(features.estimated_cardinality); // DuckDB's estimate

	// Remaining features are padding (already initialized to 0.0)
	D_ASSERT(idx <= FEATURE_VECTOR_SIZE);

	return feature_vec;
}

idx_t RLModelInterface::GetCardinalityEstimate(const OperatorFeatures &features) {
	if (!enabled) {
		return 0; // Don't override
	}

	// Print all features received by the model
	Printer::Print(features.ToString());

	// Convert features to vector
	auto feature_vec = FeaturesToVector(features);

	// Print feature vector for debugging
	Printer::Print("[RL MODEL] ========== FEATURE VECTOR ==========\n");
	Printer::Print("[RL MODEL] Feature vector size: " + std::to_string(feature_vec.size()) + "\n");

	// Print first 46 features (actual features, not padding)
	std::string vec_str = "[RL MODEL] Vector values: [";
	idx_t num_to_print = feature_vec.size() < 46 ? feature_vec.size() : 46;
	for (idx_t i = 0; i < num_to_print; i++) {
		if (i > 0) vec_str += ", ";
		vec_str += std::to_string(feature_vec[i]);
	}
	vec_str += "]\n";
	Printer::Print(vec_str);

	// Print non-zero features with labels
	Printer::Print("[RL MODEL] Non-zero features:\n");
	idx_t idx = 0;

	// Operator type (0-9)
	if (feature_vec[0] > 0) Printer::Print("[RL MODEL]   [0] Operator=GET: 1.0\n");
	if (feature_vec[1] > 0) Printer::Print("[RL MODEL]   [1] Operator=JOIN: 1.0\n");
	if (feature_vec[2] > 0) Printer::Print("[RL MODEL]   [2] Operator=FILTER: 1.0\n");
	if (feature_vec[3] > 0) Printer::Print("[RL MODEL]   [3] Operator=AGGREGATE: 1.0\n");
	if (feature_vec[9] > 0) Printer::Print("[RL MODEL]   [9] Operator=OTHER: 1.0\n");
	idx = 10;

	// Table scan features (10-17)
	if (feature_vec[10] > 0) Printer::Print("[RL MODEL]   [10] log(base_cardinality): " + std::to_string(feature_vec[10]) + "\n");
	if (feature_vec[11] > 0) Printer::Print("[RL MODEL]   [11] num_table_filters: " + std::to_string(feature_vec[11]) + "\n");
	if (feature_vec[12] > 0) Printer::Print("[RL MODEL]   [12] filter_selectivity: " + std::to_string(feature_vec[12]) + "\n");
	if (feature_vec[13] > 0) Printer::Print("[RL MODEL]   [13] used_default_selectivity: " + std::to_string(feature_vec[13]) + "\n");
	if (feature_vec[14] > 0) Printer::Print("[RL MODEL]   [14] num_filter_types: " + std::to_string(feature_vec[14]) + "\n");
	if (feature_vec[15] > 0) Printer::Print("[RL MODEL]   [15] avg_distinct_ratio: " + std::to_string(feature_vec[15]) + "\n");
	if (feature_vec[16] > 0) Printer::Print("[RL MODEL]   [16] max_distinct_ratio: " + std::to_string(feature_vec[16]) + "\n");
	if (feature_vec[17] > 0) Printer::Print("[RL MODEL]   [17] min_distinct_ratio: " + std::to_string(feature_vec[17]) + "\n");

	// Join features (18-38)
	if (feature_vec[18] > 0) Printer::Print("[RL MODEL]   [18] log(left_cardinality): " + std::to_string(feature_vec[18]) + "\n");
	if (feature_vec[19] > 0) Printer::Print("[RL MODEL]   [19] log(right_cardinality): " + std::to_string(feature_vec[19]) + "\n");
	if (feature_vec[20] > 0) Printer::Print("[RL MODEL]   [20] log(tdom_value): " + std::to_string(feature_vec[20]) + "\n");
	if (feature_vec[21] > 0) Printer::Print("[RL MODEL]   [21] tdom_from_hll: " + std::to_string(feature_vec[21]) + "\n");

	// Join type (22-26)
	if (feature_vec[22] > 0) Printer::Print("[RL MODEL]   [22] join_type=INNER: 1.0\n");
	if (feature_vec[23] > 0) Printer::Print("[RL MODEL]   [23] join_type=LEFT: 1.0\n");
	if (feature_vec[24] > 0) Printer::Print("[RL MODEL]   [24] join_type=RIGHT: 1.0\n");
	if (feature_vec[25] > 0) Printer::Print("[RL MODEL]   [25] join_type=SEMI: 1.0\n");
	if (feature_vec[26] > 0) Printer::Print("[RL MODEL]   [26] join_type=ANTI: 1.0\n");

	// Comparison type (27-32)
	if (feature_vec[27] > 0) Printer::Print("[RL MODEL]   [27] comparison=EQUAL: 1.0\n");
	if (feature_vec[28] > 0) Printer::Print("[RL MODEL]   [28] comparison=LT: 1.0\n");
	if (feature_vec[29] > 0) Printer::Print("[RL MODEL]   [29] comparison=GT: 1.0\n");
	if (feature_vec[30] > 0) Printer::Print("[RL MODEL]   [30] comparison=LTE: 1.0\n");
	if (feature_vec[31] > 0) Printer::Print("[RL MODEL]   [31] comparison=GTE: 1.0\n");
	if (feature_vec[32] > 0) Printer::Print("[RL MODEL]   [32] comparison=NEQ: 1.0\n");

	// More join features (33-38)
	if (feature_vec[33] > 0) Printer::Print("[RL MODEL]   [33] log(extra_ratio): " + std::to_string(feature_vec[33]) + "\n");
	if (feature_vec[34] > 0) Printer::Print("[RL MODEL]   [34] log(numerator): " + std::to_string(feature_vec[34]) + "\n");
	if (feature_vec[35] > 0) Printer::Print("[RL MODEL]   [35] log(denominator): " + std::to_string(feature_vec[35]) + "\n");
	if (feature_vec[36] > 0) Printer::Print("[RL MODEL]   [36] num_relations: " + std::to_string(feature_vec[36]) + "\n");
	if (feature_vec[37] > 0) Printer::Print("[RL MODEL]   [37] log(left_denominator): " + std::to_string(feature_vec[37]) + "\n");
	if (feature_vec[38] > 0) Printer::Print("[RL MODEL]   [38] log(right_denominator): " + std::to_string(feature_vec[38]) + "\n");

	// Aggregate features (39-42)
	if (feature_vec[39] > 0) Printer::Print("[RL MODEL]   [39] log(input_card_aggregate): " + std::to_string(feature_vec[39]) + "\n");
	if (feature_vec[40] > 0) Printer::Print("[RL MODEL]   [40] num_group_by_cols: " + std::to_string(feature_vec[40]) + "\n");
	if (feature_vec[41] > 0) Printer::Print("[RL MODEL]   [41] num_agg_functions: " + std::to_string(feature_vec[41]) + "\n");
	if (feature_vec[42] > 0) Printer::Print("[RL MODEL]   [42] num_grouping_sets: " + std::to_string(feature_vec[42]) + "\n");

	// Filter features (43-44)
	if (feature_vec[43] > 0) Printer::Print("[RL MODEL]   [43] log(input_card_filter): " + std::to_string(feature_vec[43]) + "\n");
	if (feature_vec[44] > 0) Printer::Print("[RL MODEL]   [44] num_filters: " + std::to_string(feature_vec[44]) + "\n");

	// Context feature (45)
	if (feature_vec[45] > 0) Printer::Print("[RL MODEL]   [45] log(duckdb_estimate): " + std::to_string(feature_vec[45]) + "\n");

	Printer::Print("[RL MODEL] ==========================================\n");

	// TODO: Feed feature_vec to ML model and get prediction

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
