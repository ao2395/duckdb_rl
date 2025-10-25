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

private:
	bool initialized;

	// Model architecture: Input(64) -> Hidden1(128) -> Hidden2(64) -> Output(1)
	static constexpr idx_t INPUT_SIZE = 64;
	static constexpr idx_t HIDDEN1_SIZE = 128;
	static constexpr idx_t HIDDEN2_SIZE = 64;
	static constexpr idx_t OUTPUT_SIZE = 1;

	// Learning rate for gradient descent
	double learning_rate;

	// Weight matrices and biases
	vector<vector<double>> weights_input_hidden1;   // 64 x 128
	vector<double> bias_hidden1;                     // 128
	vector<vector<double>> weights_hidden1_hidden2;  // 128 x 64
	vector<double> bias_hidden2;                     // 64
	vector<vector<double>> weights_hidden2_output;   // 64 x 1
	vector<double> bias_output;                      // 1

	// Activations (stored during forward pass for backprop)
	vector<double> hidden1_activations;
	vector<double> hidden2_activations;
	double output_activation;

	// Helper functions
	void InitializeWeights();
	double ReLU(double x) const;
	double ReLUDerivative(double x) const;
	vector<double> MatrixVectorMultiply(const vector<vector<double>> &matrix, const vector<double> &vec) const;
	void AddBias(vector<double> &vec, const vector<double> &bias) const;
	void ApplyReLU(vector<double> &vec) const;

	// Forward and backward pass
	double ForwardPass(const vector<double> &features);
	void BackwardPass(const vector<double> &features, double error);
};

} // namespace duckdb
