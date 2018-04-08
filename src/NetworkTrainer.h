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
	NetworkTrainer(int num_epochs_new, int samples_per_batch_new, double learning_rate_new, double test_data_frac_new, int bptt_truncate_new, bool grad_check_new, double grad_check_h_new, double grad_check_threshold_new, std::shared_ptr<NetworkType> network_new)
		: num_epochs(num_epochs_new), samples_per_batch(samples_per_batch_new), learning_rate(learning_rate_new), test_data_frac(test_data_frac_new), bptt_truncate(bptt_truncate_new), grad_check(grad_check_new), grad_check_h(grad_check_h_new), grad_check_threshold(grad_check_threshold_new), network(network_new) {

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
		int num_batches_per_epoch = std::floor(training_examples.size() / samples_per_batch);
		if(num_batches_per_epoch == 0) { num_batches_per_epoch = 1; }
		for(int epoch = 0; epoch < num_epochs; epoch++) {
			BOOST_LOG_TRIVIAL(info) << "Training epoch " << epoch << "...";

			// Randomize training example #s
			BOOST_LOG_TRIVIAL(info) << "Randomizing training examples...";
			std::vector<int> examples_remaining_for_epoch(training_examples.size());
			std::iota(examples_remaining_for_epoch.begin(), examples_remaining_for_epoch.end(), 0);
			std::random_shuffle(examples_remaining_for_epoch.begin(), examples_remaining_for_epoch.end());

			// Iterate through batches and take examples each time
			for(int batch = 0; batch < num_batches_per_epoch; batch++) {
				BOOST_LOG_TRIVIAL(info) << "Training epoch " << epoch << ", batch " << batch << "...";

				// If on last batch, take rest of examples.  Otherwise, take normal size.
				int num_examples_in_batch;
				if(batch == num_batches_per_epoch - 1) { num_examples_in_batch = examples_remaining_for_epoch.size(); }
				else { num_examples_in_batch = samples_per_batch; }

				// Calculate avg gradients
				BOOST_LOG_TRIVIAL(info) << "Training epoch " << epoch << ", batch " << batch << ", calculating gradients...";
				arma::mat dCdW(network->getW().n_rows, network->getW().n_cols, arma::fill::zeros);
				arma::mat dCdU(network->getU().n_rows, network->getU().n_cols, arma::fill::zeros);
				arma::mat dCdV(network->getV().n_rows, network->getV().n_cols, arma::fill::zeros);
				std::unique_ptr<arma::mat> tmpW;
				std::unique_ptr<arma::mat> tmpU;
				std::unique_ptr<arma::mat> tmpV;
				for(int ex = 0; ex < num_examples_in_batch; ex++) {
					int exNum = training_examples[examples_remaining_for_epoch.back()];
					examples_remaining_for_epoch.pop_back();

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
					dCdW += *tmpW;
					dCdU += *tmpU;
					dCdV += *tmpV;

					if(grad_check){
						checkGrad(*tmpW, "W", x[exNum], y[exNum]);
					}
				}
				dCdW /= num_examples_in_batch;
				dCdU /= num_examples_in_batch;
				dCdV /= num_examples_in_batch;

				// Update weights per gradients
				BOOST_LOG_TRIVIAL(info) << "Training epoch " << epoch << ", batch " << batch << ", updating weights...";
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
			double percent = ProgressEvaluator::evalPercentWordsCorrect(*network, predict, test_examples_correct);
			BOOST_LOG_TRIVIAL(info) << percent << " fraction correct!";
		}
	}

	void checkGrad(const arma::mat& grad, const std::string& param, const Sentence& x, const Sentence& y) {
		BOOST_LOG_TRIVIAL(info) << "Checking gradients...";

		int n_rows;
		int n_cols;
		if(param == "W") {
			n_rows = network->getW().n_rows;
			n_cols = network->getW().n_cols;
		} else if(param == "U") {
			n_rows = network->getU().n_rows;
			n_cols = network->getU().n_cols;
		} else if(param == "V") {
			n_rows = network->getV().n_rows;
			n_cols = network->getV().n_cols;
		} else {
			throw std::runtime_error("Invalid param name!");
		}

		for(int row = 0; row < n_rows; row++) {
			for(int col = 0; col < n_cols; col++) {
				// Get cost for +h
				if(param == "W") {
					network->updateWVal(row, col, grad_check_h);
				} else if(param == "U") {
					network->updateUVal(row, col, grad_check_h);
				} else if(param == "V") {
					network->updateVVal(row, col, grad_check_h);
				} else {
					throw std::runtime_error("Invalid param name!");
				}
				std::unique_ptr<arma::mat> ss;
				std::unique_ptr<Sentence> out;
				network->feedForward(x, ss, out);
				double costP = ActivationLossConfig::evalCost(y, *out);

				// Get cost for -h
				if(param == "W") {
					network->updateWVal(row, col, -2 * grad_check_h);
				} else if(param == "U") {
					network->updateUVal(row, col, -2 * grad_check_h);
				} else if(param == "V") {
					network->updateVVal(row, col, -2 * grad_check_h);
				} else {
					throw std::runtime_error("Invalid param name!");
				}
				network->feedForward(x, ss, out);
				double costN = ActivationLossConfig::evalCost(y, *out);

				// Reset to orig value
				if(param == "W") {
					network->updateWVal(row, col, grad_check_h);
				} else if(param == "U") {
					network->updateUVal(row, col, grad_check_h);
				} else if(param == "V") {
					network->updateVVal(row, col, grad_check_h);
				} else {
					throw std::runtime_error("Invalid param name!");
				}

				// Calc gradient and compare to actual value
				double gradientEst = (costP - costN) / (2.0 * grad_check_h);
				double rel_error = std::abs(gradientEst - grad(row, col)) / (std::abs(gradientEst) + std::abs(grad(row, col)));
				if(rel_error > grad_check_threshold) {
					std::cout << ("Grad check failed!  Correct: " + std::to_string(gradientEst) + " Calculated: " + std::to_string(grad(row, col))) << std::endl;
				}
			}
		}
	}
private:
	int num_epochs;
	int samples_per_batch;
	double learning_rate;
	double test_data_frac;
	int bptt_truncate;
	bool grad_check;
	double grad_check_h;
	double grad_check_threshold;
	std::shared_ptr<NetworkType> network;
};

#endif /* SRC_NETWORKTRAINER_H_ */
