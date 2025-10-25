#include "duckdb/execution/operator/filter/physical_filter.hpp"
#include "duckdb/execution/operator/projection/physical_projection.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/optimizer/matcher/expression_matcher.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/main/rl_model_interface.hpp"

namespace duckdb {

PhysicalOperator &PhysicalPlanGenerator::CreatePlan(LogicalFilter &op) {
	D_ASSERT(op.children.size() == 1);
	reference<PhysicalOperator> plan = CreatePlan(*op.children[0]);

	// RL MODEL INFERENCE: After child is created, extract features and get estimate
	RLModelInterface rl_model(context);
	auto features = rl_model.ExtractFeatures(op, context);
	auto rl_estimate = rl_model.GetCardinalityEstimate(features);
	// For now, we don't override - just print features (rl_estimate will be 0)
	if (rl_estimate > 0) {
		op.estimated_cardinality = rl_estimate;
	}

	if (!op.expressions.empty()) {
		D_ASSERT(!plan.get().GetTypes().empty());
		// create a filter if there is anything to filter
		auto &filter = Make<PhysicalFilter>(plan.get().GetTypes(), std::move(op.expressions), op.estimated_cardinality);
		filter.children.push_back(plan);
		plan = filter;
	}
	if (op.HasProjectionMap()) {
		// there is a projection map, generate a physical projection
		vector<unique_ptr<Expression>> select_list;
		for (idx_t i = 0; i < op.projection_map.size(); i++) {
			select_list.push_back(make_uniq<BoundReferenceExpression>(op.types[i], op.projection_map[i]));
		}
		auto &proj = Make<PhysicalProjection>(op.types, std::move(select_list), op.estimated_cardinality);
		proj.children.push_back(plan);
		plan = proj;
	}
	return plan;
}

} // namespace duckdb
