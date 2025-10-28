//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/main/rl_training_thread.cpp
//
//===----------------------------------------------------------------------===//

#include "duckdb/main/rl_training_thread.hpp"
#include "duckdb/main/rl_cardinality_model.hpp"
#include "duckdb/common/printer.hpp"
#include <chrono>

namespace duckdb {

RLTrainingThread::RLTrainingThread(RLCardinalityModel &model, RLTrainingBuffer &buffer)
    : model(model), buffer(buffer), should_stop(false), is_running(false),
      total_updates(0), running_loss_sum(0.0), loss_count(0) {
}

RLTrainingThread::~RLTrainingThread() {
	Stop();
}

void RLTrainingThread::Start(const RLTrainingConfig &cfg) {
	if (is_running) {
		Printer::Print("[RL TRAINING THREAD] Already running\n");
		return;
	}

	config = cfg;
	should_stop = false;

	Printer::Print("[RL TRAINING THREAD] Starting background training with config:\n");
	Printer::Print("  Batch size: " + std::to_string(config.batch_size) + "\n");
	Printer::Print("  Min buffer size: " + std::to_string(config.min_buffer_size) + "\n");
	Printer::Print("  Training interval: " + std::to_string(config.training_interval_ms) + "ms\n");
	Printer::Print("  Max iterations per cycle: " + std::to_string(config.max_iterations_per_cycle) + "\n");

	// Start background thread
	training_thread = std::thread(&RLTrainingThread::TrainingLoop, this);
	is_running = true;
}

void RLTrainingThread::Stop() {
	if (!is_running) {
		return;
	}

	Printer::Print("[RL TRAINING THREAD] Stopping background training...\n");
	should_stop = true;
	training_cv.notify_all();

	if (training_thread.joinable()) {
		training_thread.join();
	}

	is_running = false;
	Printer::Print("[RL TRAINING THREAD] Stopped. Total updates: " + std::to_string(total_updates.load()) + "\n");
}

bool RLTrainingThread::IsRunning() const {
	return is_running;
}

idx_t RLTrainingThread::GetTotalUpdates() const {
	return total_updates.load();
}

double RLTrainingThread::GetAverageTrainingLoss() const {
	idx_t count = loss_count.load();
	if (count == 0) {
		return 0.0;
	}
	return running_loss_sum.load() / count;
}

void RLTrainingThread::TrainingLoop() {
	Printer::Print("[RL TRAINING THREAD] Training loop started\n");

	while (!should_stop) {
		// Wait for training interval or until stopped
		std::unique_lock<mutex> lock(training_mutex);
		training_cv.wait_for(lock, std::chrono::milliseconds(config.training_interval_ms),
		                      [this]() { return should_stop.load(); });

		if (should_stop) {
			break;
		}

		// Check if we have enough samples to train
		if (buffer.Size() < config.min_buffer_size) {
			continue;
		}

		// Perform training
		TrainBatch();
	}

	Printer::Print("[RL TRAINING THREAD] Training loop exiting\n");
}

void RLTrainingThread::TrainBatch() {
	// Get a batch of samples from the buffer
	auto batch = buffer.GetBatch(config.batch_size);

	if (batch.empty()) {
		return;
	}

	double total_loss = 0.0;

	// Perform multiple gradient updates on this batch
	for (idx_t iteration = 0; iteration < config.max_iterations_per_cycle; iteration++) {
		for (const auto &sample : batch) {
			// Update model with this sample
			model.Update(sample.features, sample.actual_cardinality, sample.predicted_cardinality);

			// Track loss (Q-error)
			total_loss += sample.q_error;
		}

		total_updates += batch.size();
	}

	// Update statistics
	double avg_loss = total_loss / (batch.size() * config.max_iterations_per_cycle);
	// Atomics don't support +=, so use fetch_add or store with load
	running_loss_sum.store(running_loss_sum.load() + avg_loss);
	loss_count++;

	// Log training progress periodically
	if (total_updates % 1000 == 0) {
		Printer::Print("[RL TRAINING THREAD] Updates: " + std::to_string(total_updates.load()) +
		               ", Avg Loss: " + std::to_string(GetAverageTrainingLoss()) +
		               ", Buffer Size: " + std::to_string(buffer.Size()) +
		               ", Avg Buffer Q-error: " + std::to_string(buffer.AverageQError()) + "\n");
	}
}

} // namespace duckdb
