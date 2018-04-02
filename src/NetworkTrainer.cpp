/*
 * NetworkTrainer.cpp
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#include "NetworkTrainer.h"

template<class NetworkType, class ProgressEvaluator, class CostFunction>
NetworkTrainer<NetworkType, ProgressEvaluator, CostFunction>::NetworkTrainer(int num_epochs_new, int samples_per_epoch_new, double learning_rate_new, double test_data_frac_new, int bptt_truncate_new, std::shared_ptr<NetworkType> network_new)
	: num_epochs(num_epochs_new), samples_per_epoch(samples_per_epoch_new), learning_rate(learning_rate_new), test_data_frac(test_data_frac_new), bptt_truncate(bptt_truncate_new), network(network_new) {

}

template<class NetworkType, class ProgressEvaluator, class CostFunction>
void NetworkTrainer<NetworkType, ProgressEvaluator, CostFunction>::train(const SentenceList& x, const SentenceList& y) {
	// Figure out how many test/training examples we will have
	int num_test_exs = x.size() * test_data_frac;
	int num_training_exs = x.size() - num_test_exs;
	std::vector<int> test_examples(num_test_exs);
	std::vector<int> training_examples(num_training_exs);

	// Randomly assign test/training examples
	std::vector<int> randomized_examples(x.size());
	std::iota(randomized_examples.begin(), randomized_examples.end(), 0);
	std::random_shuffle(randomized_examples.begin(), randomized_examples.end());
	for(int random_ex_num = 0; random_ex_num < x.size(); random_ex_num++) {

		if(random_ex_num < num_test_exs) {
			test_examples[random_ex_num] = randomized_examples[random_ex_num];
		} else {
			training_examples[random_ex_num - num_test_exs] = randomized_examples[random_ex_num];
		}
	}

	// Copy over test Y values to a separate array (we will need this during progress eval)
	SentenceList test_examples_correct(num_test_exs);
	for(int ex_num = 0; ex_num < num_test_exs; ex_num++) {
		test_examples_correct[ex_num] = y[test_examples[ex_num]];
	}

	// Iterate through training epochs
	int num_batches_per_epoch = std::floor(training_examples.size() / samples_per_epoch);
	if(num_batches_per_epoch == 0) { num_batches_per_epoch = 1; }
	for(int epoch = 0; epoch < num_epochs; epoch++) {
		// Randomize training example #s
		std::vector<int> examples_remaining_for_epoch(training_examples.size());
		std::iota(examples_remaining_for_epoch.begin(), examples_remaining_for_epoch.end(), 0);
		std::random_shuffle(examples_remaining_for_epoch.begin(), examples_remaining_for_epoch.end());

		// Iterate through batches and take examples each time
		for(int batch = 0; batch < num_batches_per_epoch; batch++) {
			// If on last batch, take rest of examples.  Otherwise, take normal size.
			int num_examples_in_batch;
			if(batch == num_batches_per_epoch - 1) { num_examples_in_batch = examples_remaining_for_epoch.size(); }
			else { num_examples_in_batch = samples_per_epoch; }

			// Calculate avg gradients
			arma::mat dCdW(network->getW().n_rows, network->getW().n_cols, arma::fill::zeros);
			arma::mat dCdU(network->getU().n_rows, network->getU().n_cols, arma::fill::zeros);
			arma::mat dCdV(network->getV().n_rows, network->getV().n_cols, arma::fill::zeros);
			for(int ex = 0; ex < num_examples_in_batch; ex++) {
				int exNum = training_examples[examples_remaining_for_epoch.back()];
				examples_remaining_for_epoch.pop_back();

				std::unique_ptr<arma::mat> saved_states;
				std::unique_ptr<arma::mat> outputs;
				network->feedForward(x[exNum], saved_states, outputs);
				NetworkType::ActivationLossConfig::addGradients(
					network,
					bptt_truncate,
					x[exNum],
					y[exNum],
					*saved_states,
					*outputs,
					dCdW,
					dCdU,
					dCdV
				);
			}
			dCdW /= num_examples_in_batch;
			dCdU /= num_examples_in_batch;
			dCdV /= num_examples_in_batch;

			// Update weights per gradients
			network->setW(network->getW() - (dCdW * learning_rate));
			network->setU(network->getU() - (dCdU * learning_rate));
			network->setV(network->getV() - (dCdV * learning_rate));
		}

		// Eval progress
		BOOST_LOG_TRIVIAL(info) << "Finished epoch " << epoch << ".  Evaluating progress...";
		SentenceList predict(num_test_exs);
		std::unique_ptr<arma::mat> tmp_ptr_a;
		std::unique_ptr<arma::mat> tmp_outputs;
		for(int ex_num = 0; ex_num < num_test_exs; ex_num++) {
			std::unique_ptr<arma::mat> outputs;
			network->feedForward(x[test_examples[ex_num]], tmp_ptr_a, tmp_outputs);
			predict[ex_num] = std::move(*tmp_outputs);
		}
		ProgressEvaluator::eval(network, predict, test_examples_correct);
	}
}
