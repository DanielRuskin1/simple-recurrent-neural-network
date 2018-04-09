/*
 * GradientChecker.h
 *
 *  Created on: Apr 8, 2018
 *      Author: danielruskin
 */

#ifndef SRC_GRADIENTCHECKER_H_
#define SRC_GRADIENTCHECKER_H_

#include <boost/log/trivial.hpp>

template<class NetworkType, class ActivationLossConfig>
class GradientChecker {
	GradientChecker(double h_new, double error_threshold_new)
		: h(h_new), error_threshold(error_threshold_new) {

	}

	void checkGradients(const std::string& output_prefix, const NetworkType& network, const Sentence& x, const Sentence& y) {
		// TODO
	}
private:
	double h;
	double error_threshold;
};

#endif /* SRC_GRADIENTCHECKER_H_ */
