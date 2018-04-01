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

template<class NetworkType, class ProgressEvaluator, class CostFunction>
class NetworkTrainer {
public:
	NetworkTrainer(int num_epochs_new, int samples_per_epoch_new, double learning_rate_new, double test_data_frac_new, std::shared_ptr<NetworkType> network_new);

	void train(const SentenceList& sl);
private:
	int num_epochs;
	int samples_per_epoch;
	double learning_rate;
	double test_data_frac;
	std::shared_ptr<NetworkType> network;

	// Accepts pre-set references, and will add gradients to those references.
	void addGradients(const Sentence& input, const arma::mat& saved_states, const arma::mat& outputs, arma::mat& out_dCdW, arma::mat& out_dCdU, arma::mat& out_dCdV);
};

#endif /* SRC_NETWORKTRAINER_H_ */
