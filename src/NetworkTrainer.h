/*
 * NetworkTrainer.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_NETWORKTRAINER_H_
#define SRC_NETWORKTRAINER_H_

#include <memory>
#include "Utils.h"
#include <boost/log/trivial.hpp>

template<class NetworkType, class ActivationLossConfig, class ProgressEvaluator>
class NetworkTrainer {
public:
	NetworkTrainer(int num_epochs_new, int samples_per_batch_new, double learning_rate_new, double test_data_frac_new, int bptt_truncate_new, std::shared_ptr<NetworkType> network_new)
		: num_epochs(num_epochs_new), samples_per_batch(samples_per_batch_new), learning_rate(learning_rate_new), test_data_frac(test_data_frac_new), bptt_truncate(bptt_truncate_new), network(network_new) {

	}

	void train(const SentenceList& x, const SentenceList& y) {
		// Figure out how many test/training examples we will have
		int num_test_exs = std::ceil(x.size() * test_data_frac);
		int num_training_exs = x.size() - num_test_exs;
		std::vector<int> test_examples(num_test_exs);
		std::vector<int> training_examples(num_training_exs);

		// Randomly assign test/training examples
		BOOST_LOG_TRIVIAL(info) << "Assigning examples to test/training set...";
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
		for(int epoch = 0; epoch < num_epochs; epoch++) {
			BOOST_LOG_TRIVIAL(info) << "Training epoch " << epoch << "...";

			// Randomize training example #s
			BOOST_LOG_TRIVIAL(info) << "Randomizing training examples...";
			std::vector<int> examples_for_epoch(training_examples.size());
			std::iota(examples_for_epoch.begin(), examples_for_epoch.end(), 0);
			std::random_shuffle(examples_for_epoch.begin(), examples_for_epoch.end());

			// Iterate through batches and take examples each time
			int at_epoch_example = 0;
			int batch = 0;
			while(at_epoch_example < examples_for_epoch.size()) {
				BOOST_LOG_TRIVIAL(info) << "Training epoch " << epoch << ", batch " << batch << "...";

				// If on last batch, take rest of examples.  Otherwise, take normal size.
				int unused_examples_for_epoch = examples_for_epoch.size() - at_epoch_example;
				int num_examples_in_batch = std::min(samples_per_batch, unused_examples_for_epoch);

				// Calculate avg gradients
				BOOST_LOG_TRIVIAL(info) << "Calculating gradients...";
				arma::mat dCdW(network->getW().n_rows, network->getW().n_cols, arma::fill::zeros);
				arma::mat dCdU(network->getU().n_rows, network->getU().n_cols, arma::fill::zeros);
				arma::mat dCdV(network->getV().n_rows, network->getV().n_cols, arma::fill::zeros);

				#pragma omp parallel
				#pragma omp for
				for(int ex = 0; ex < num_examples_in_batch; ex++) {
					std::unique_ptr<arma::mat> tmpW;
					std::unique_ptr<arma::mat> tmpU;
					std::unique_ptr<arma::mat> tmpV;

					int exNum = training_examples[examples_for_epoch[at_epoch_example + ex]];

					std::unique_ptr<arma::mat> saved_states;
					std::unique_ptr<arma::mat> outputs;
					network->feedForward(x[exNum], saved_states, outputs);
					ActivationLossConfig::setGradients(
						*network,
						bptt_truncate,
						x[exNum],
						y[exNum],
						*saved_states,
						*outputs,
						tmpW,
						tmpU,
						tmpV
					);

					#pragma omp critical
					{
						dCdW += *tmpW;
						dCdU += *tmpU;
						dCdV += *tmpV;
					}
				}
				dCdW /= num_examples_in_batch;
				dCdU /= num_examples_in_batch;
				dCdV /= num_examples_in_batch;

				// Update weights per gradients
				BOOST_LOG_TRIVIAL(info) << "Updating weights...";
				network->setW(network->getW() - (dCdW * learning_rate));
				network->setU(network->getU() - (dCdU * learning_rate));
				network->setV(network->getV() - (dCdV * learning_rate));

				at_epoch_example += num_examples_in_batch;
				batch++;
			}

			// Eval progress
			BOOST_LOG_TRIVIAL(info) << "Evaluating progress...";
			SentenceList predict(num_test_exs);
			std::unique_ptr<arma::mat> tmp_ptr_a;
			std::unique_ptr<arma::mat> tmp_outputs;
			for(int ex_num = 0; ex_num < num_test_exs; ex_num++) {
				std::unique_ptr<arma::mat> outputs;
				network->feedForward(x[test_examples[ex_num]], tmp_ptr_a, tmp_outputs);
				predict[ex_num] = std::move(*tmp_outputs);
			}
			double percent = ProgressEvaluator::evalPercentWordsCorrect(*network, predict, test_examples_correct);
			BOOST_LOG_TRIVIAL(info) << percent << " fraction correct!";
		}
	}
private:
	int num_epochs;
	int samples_per_batch;
	double learning_rate;
	double test_data_frac;
	int bptt_truncate;
	std::shared_ptr<NetworkType> network;
};

#endif /* SRC_NETWORKTRAINER_H_ */
