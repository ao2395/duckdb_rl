//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_cardinality_model.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/mutex.hpp"

namespace duckdb {

//! Multi-layer Perceptron for online reinforcement learning cardinality estimation
//! Learns from query execution feedback (Q-error) to improve estimates over time
//! Singleton pattern - one model instance shared across all queries
class RLCardinalityModel {
public:
	//! Get the singleton instance
	static RLCardinalityModel &Get();

	// Delete copy constructor and assignment operator
	RLCardinalityModel(const RLCardinalityModel &) = delete;
	RLCardinalityModel &operator=(const RLCardinalityModel &) = delete;

private:
	RLCardinalityModel();
	~RLCardinalityModel();

public:

	//! Perform inference: takes feature vector and returns estimated cardinality
	//! Input: 64-dimensional feature vector
	//! Output: predicted cardinality (NOT log - we convert internally)
	double Predict(const vector<double> &features);

	//! Train the model with actual cardinality feedback (online RL update)
	//! Uses gradient descent to minimize Q-error
	void Update(const vector<double> &features, idx_t actual_cardinality, idx_t predicted_cardinality);

	//! Save/load model weights (optional, for checkpointing)
	void SaveWeights(const string &model_path);
	void LoadWeights(const string &model_path);

	//! Check if model is ready for inference
	bool IsReady() const {
		return initialized;
	}

	//! Reset model weights to initial state (in case of instability)
	void ResetWeights();

private:
	bool initialized;

	// Thread safety: Separate locks for read (inference) and write (training)
	// Allows concurrent readers, exclusive writer
	mutable mutex model_lock;

	// Model architecture: Input(64) -> Hidden1(128) -> Hidden2(128) -> Hidden3(64) -> Output(1)
	static constexpr idx_t INPUT_SIZE = 64;
	static constexpr idx_t HIDDEN1_SIZE = 128;
	static constexpr idx_t HIDDEN2_SIZE = 128;
	static constexpr idx_t HIDDEN3_SIZE = 64;
	static constexpr idx_t OUTPUT_SIZE = 1;

	// Learning rate for gradient descent
	double learning_rate;

	// Weight matrices and biases (protected by model_lock)
	vector<vector<double>> weights_input_hidden1;   // 64 x 128
	vector<double> bias_hidden1;                     // 128
	vector<vector<double>> weights_hidden1_hidden2;  // 128 x 128
	vector<double> bias_hidden2;                     // 128
	vector<vector<double>> weights_hidden2_hidden3;  // 128 x 64
	vector<double> bias_hidden3;                     // 64
	vector<vector<double>> weights_hidden3_output;   // 64 x 1
	vector<double> bias_output;                      // 1

	// Helper functions
	void InitializeWeights();
	double ReLU(double x) const;
	double ReLUDerivative(double x) const;
	vector<double> MatrixVectorMultiply(const vector<vector<double>> &matrix, const vector<double> &vec) const;
	void AddBias(vector<double> &vec, const vector<double> &bias) const;
	void ApplyReLU(vector<double> &vec) const;

	// Forward and backward pass (caller must hold model_lock)
	double ForwardPassUnlocked(const vector<double> &features,
	                            vector<double> &hidden1_out,
	                            vector<double> &hidden2_out,
	                            vector<double> &hidden3_out) const;
	void BackwardPassUnlocked(const vector<double> &features,
	                           const vector<double> &hidden1_activations,
	                           const vector<double> &hidden2_activations,
	                           const vector<double> &hidden3_activations,
	                           double error);
};

} // namespace duckdb
