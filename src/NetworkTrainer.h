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

template<class NetworkType, class ProgressEvaluator, class CostFunction>
class NetworkTrainer {
public:
	NetworkTrainer(int num_epochs_new, int samples_per_epoch_new, double learning_rate_new, double test_data_frac_new, int bptt_truncate_new, std::shared_ptr<NetworkType> network_new);

	void train(const SentenceList& x, const SentenceList& y);
private:
	int num_epochs;
	int samples_per_epoch;
	double learning_rate;
	double test_data_frac;
	int bptt_truncate;
	std::shared_ptr<NetworkType> network;
};

#endif /* SRC_NETWORKTRAINER_H_ */
