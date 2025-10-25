#include "duckdb/main/rl_cardinality_model.hpp"
#include "duckdb/common/printer.hpp"
#include <cmath>
#include <random>

namespace duckdb {

RLCardinalityModel &RLCardinalityModel::Get() {
	static RLCardinalityModel instance;
	return instance;
}

RLCardinalityModel::RLCardinalityModel() : initialized(false), learning_rate(0.001) {
	Printer::Print("[RL MODEL] Initializing singleton MLP for online RL...\n");
	InitializeWeights();
	initialized = true;
	Printer::Print("[RL MODEL] MLP initialized with architecture: 64 -> 128 -> 64 -> 1\n");
}

RLCardinalityModel::~RLCardinalityModel() {
}

void RLCardinalityModel::InitializeWeights() {
	// Xavier/He initialization for better convergence
	std::random_device rd;
	std::mt19937 gen(rd());

	// Initialize weights_input_hidden1 (64 x 128)
	double std_dev1 = std::sqrt(2.0 / INPUT_SIZE);
	std::normal_distribution<double> dist1(0.0, std_dev1);
	weights_input_hidden1.resize(HIDDEN1_SIZE);
	for (idx_t i = 0; i < HIDDEN1_SIZE; i++) {
		weights_input_hidden1[i].resize(INPUT_SIZE);
		for (idx_t j = 0; j < INPUT_SIZE; j++) {
			weights_input_hidden1[i][j] = dist1(gen);
		}
	}
	bias_hidden1.resize(HIDDEN1_SIZE, 0.0);

	// Initialize weights_hidden1_hidden2 (128 x 64)
	double std_dev2 = std::sqrt(2.0 / HIDDEN1_SIZE);
	std::normal_distribution<double> dist2(0.0, std_dev2);
	weights_hidden1_hidden2.resize(HIDDEN2_SIZE);
	for (idx_t i = 0; i < HIDDEN2_SIZE; i++) {
		weights_hidden1_hidden2[i].resize(HIDDEN1_SIZE);
		for (idx_t j = 0; j < HIDDEN1_SIZE; j++) {
			weights_hidden1_hidden2[i][j] = dist2(gen);
		}
	}
	bias_hidden2.resize(HIDDEN2_SIZE, 0.0);

	// Initialize weights_hidden2_output (64 x 1)
	double std_dev3 = std::sqrt(2.0 / HIDDEN2_SIZE);
	std::normal_distribution<double> dist3(0.0, std_dev3);
	weights_hidden2_output.resize(OUTPUT_SIZE);
	for (idx_t i = 0; i < OUTPUT_SIZE; i++) {
		weights_hidden2_output[i].resize(HIDDEN2_SIZE);
		for (idx_t j = 0; j < HIDDEN2_SIZE; j++) {
			weights_hidden2_output[i][j] = dist3(gen);
		}
	}
	bias_output.resize(OUTPUT_SIZE, 0.0);

	// Initialize activation storage
	hidden1_activations.resize(HIDDEN1_SIZE);
	hidden2_activations.resize(HIDDEN2_SIZE);
}

double RLCardinalityModel::ReLU(double x) const {
	return x > 0.0 ? x : 0.0;
}

double RLCardinalityModel::ReLUDerivative(double x) const {
	return x > 0.0 ? 1.0 : 0.0;
}

vector<double> RLCardinalityModel::MatrixVectorMultiply(const vector<vector<double>> &matrix,
                                                         const vector<double> &vec) const {
	vector<double> result(matrix.size(), 0.0);
	for (idx_t i = 0; i < matrix.size(); i++) {
		for (idx_t j = 0; j < vec.size(); j++) {
			result[i] += matrix[i][j] * vec[j];
		}
	}
	return result;
}

void RLCardinalityModel::AddBias(vector<double> &vec, const vector<double> &bias) const {
	for (idx_t i = 0; i < vec.size(); i++) {
		vec[i] += bias[i];
	}
}

void RLCardinalityModel::ApplyReLU(vector<double> &vec) const {
	for (idx_t i = 0; i < vec.size(); i++) {
		vec[i] = ReLU(vec[i]);
	}
}

double RLCardinalityModel::ForwardPass(const vector<double> &features) {
	// Layer 1: Input -> Hidden1
	hidden1_activations = MatrixVectorMultiply(weights_input_hidden1, features);
	AddBias(hidden1_activations, bias_hidden1);
	ApplyReLU(hidden1_activations);

	// Layer 2: Hidden1 -> Hidden2
	hidden2_activations = MatrixVectorMultiply(weights_hidden1_hidden2, hidden1_activations);
	AddBias(hidden2_activations, bias_hidden2);
	ApplyReLU(hidden2_activations);

	// Layer 3: Hidden2 -> Output
	auto output = MatrixVectorMultiply(weights_hidden2_output, hidden2_activations);
	AddBias(output, bias_output);

	// Store output (no activation on output layer - linear)
	output_activation = output[0];

	// Return predicted log(cardinality)
	return output_activation;
}

double RLCardinalityModel::Predict(const vector<double> &features) {
	// Validate input size
	if (features.size() != INPUT_SIZE) {
		Printer::Print("[RL MODEL ERROR] Invalid feature vector size: " + std::to_string(features.size()) +
		               " (expected " + std::to_string(INPUT_SIZE) + ")\n");
		return 0.0;
	}

	if (!initialized) {
		Printer::Print("[RL MODEL] Model not initialized\n");
		return 0.0;
	}

	// // Print the feature vector (compact format)
	// Printer::Print("[RL MODEL] Feature vector: [");
	// for (idx_t i = 0; i < features.size(); i++) {
	// 	if (i > 0) Printer::Print(", ");
	// 	Printer::Print(std::to_string(features[i]));
	// }
	// Printer::Print("]\n");

	// Forward pass through the network
	double log_cardinality = ForwardPass(features);

	// Convert from log(cardinality) to cardinality
	double cardinality = std::exp(log_cardinality);

	// Clamp to reasonable range
	if (cardinality < 1.0) cardinality = 1.0;
	if (cardinality > 1e15) cardinality = 1e15;

	Printer::Print("[RL MODEL] MLP prediction: log(card)=" + std::to_string(log_cardinality) +
	               " -> card=" + std::to_string(cardinality) + "\n");

	return cardinality;
}

void RLCardinalityModel::BackwardPass(const vector<double> &features, double error) {
	// Compute gradients using backpropagation
	// Loss = MSE on log(cardinality), so error = predicted_log - actual_log

	// Output layer gradients
	vector<double> output_grad(OUTPUT_SIZE);
	output_grad[0] = error; // d(Loss)/d(output)

	// Hidden2 -> Output weight gradients
	for (idx_t i = 0; i < OUTPUT_SIZE; i++) {
		for (idx_t j = 0; j < HIDDEN2_SIZE; j++) {
			double grad = output_grad[i] * hidden2_activations[j];
			weights_hidden2_output[i][j] -= learning_rate * grad;
		}
		bias_output[i] -= learning_rate * output_grad[i];
	}

	// Backpropagate to hidden2
	vector<double> hidden2_grad(HIDDEN2_SIZE, 0.0);
	for (idx_t j = 0; j < HIDDEN2_SIZE; j++) {
		for (idx_t i = 0; i < OUTPUT_SIZE; i++) {
			hidden2_grad[j] += output_grad[i] * weights_hidden2_output[i][j];
		}
		hidden2_grad[j] *= ReLUDerivative(hidden2_activations[j]);
	}

	// Hidden1 -> Hidden2 weight gradients
	for (idx_t i = 0; i < HIDDEN2_SIZE; i++) {
		for (idx_t j = 0; j < HIDDEN1_SIZE; j++) {
			double grad = hidden2_grad[i] * hidden1_activations[j];
			weights_hidden1_hidden2[i][j] -= learning_rate * grad;
		}
		bias_hidden2[i] -= learning_rate * hidden2_grad[i];
	}

	// Backpropagate to hidden1
	vector<double> hidden1_grad(HIDDEN1_SIZE, 0.0);
	for (idx_t j = 0; j < HIDDEN1_SIZE; j++) {
		for (idx_t i = 0; i < HIDDEN2_SIZE; i++) {
			hidden1_grad[j] += hidden2_grad[i] * weights_hidden1_hidden2[i][j];
		}
		hidden1_grad[j] *= ReLUDerivative(hidden1_activations[j]);
	}

	// Input -> Hidden1 weight gradients
	for (idx_t i = 0; i < HIDDEN1_SIZE; i++) {
		for (idx_t j = 0; j < INPUT_SIZE; j++) {
			double grad = hidden1_grad[i] * features[j];
			weights_input_hidden1[i][j] -= learning_rate * grad;
		}
		bias_hidden1[i] -= learning_rate * hidden1_grad[i];
	}
}

void RLCardinalityModel::Update(const vector<double> &features, idx_t actual_cardinality,
                                 idx_t predicted_cardinality) {
	if (!initialized) {
		return;
	}

	// Compute target and prediction in log space
	double actual_log = std::log(std::max((double)actual_cardinality, 1.0));
	double predicted_log = std::log(std::max((double)predicted_cardinality, 1.0));

	// Compute error (MSE gradient)
	double error = predicted_log - actual_log;

	// Compute Q-error for logging
	double q_error = std::max(actual_cardinality / (double)std::max(predicted_cardinality, (idx_t)1),
	                          predicted_cardinality / (double)std::max(actual_cardinality, (idx_t)1));

	Printer::Print("[RL MODEL] Training update: actual=" + std::to_string(actual_cardinality) +
	               ", predicted=" + std::to_string(predicted_cardinality) +
	               ", Q-error=" + std::to_string(q_error) + "\n");

	// Backpropagation to update weights
	BackwardPass(features, error);
}

void RLCardinalityModel::SaveWeights(const string &model_path) {
	// TODO: Implement weight serialization to file
	Printer::Print("[RL MODEL] SaveWeights not yet implemented for path: " + model_path + "\n");
}

void RLCardinalityModel::LoadWeights(const string &model_path) {
	// TODO: Implement weight loading from file
	Printer::Print("[RL MODEL] LoadWeights not yet implemented for path: " + model_path + "\n");
}

} // namespace duckdb
