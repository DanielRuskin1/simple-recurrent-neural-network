/*
 * TanHActivation.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_TANHACTIVATION_H_
#define SRC_TANHACTIVATION_H_

std::unique_ptr<arma::colvec> eval(const arma::colvec& in) {
	return std::unique_ptr<arma::colvec>(new arma::colvec(arma::tanh(in)));
}

// Derivative of tanh is simply 1 - tanh^2
std::unique_ptr<arma::colvec> evalPrime(const arma::colvec& in) {
	return std::unique_ptr<arma::colvec>(new arma::colvec(1 - arma::pow(arma::tanh(in), 2)));
}

#endif /* SRC_TANHACTIVATION_H_ */
