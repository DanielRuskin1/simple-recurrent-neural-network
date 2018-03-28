/*
 * SoftmaxActivation.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_SOFTMAXACTIVATION_H_
#define SRC_SOFTMAXACTIVATION_H_

std::unique_ptr<arma::colvec> eval(const arma::colvec& in) {
	arma::colvec matExp = arma::exp(in);

	return std::unique_ptr<arma::colvec>(new arma::colvec(matExp / arma::accu(matExp)));
}

// Derivative of output i with respect to input j is:
// -i*j if i != j
// i(1 - i) if i == j
std::unique_ptr<arma::mat> evalPrime(const arma::colvec& in) {
	// Handle first case
	std::unique_ptr<arma::colvec> result(new arma::mat(arma::kron(in, arma::trans(in))));

	// Replace values to handle second case
	for(int i = 0; i < in.n_rows; i++) {
		(*result)(i, i) = in(i) * (1 - in(i));
	}

	return result;
}

#endif /* SRC_SOFTMAXACTIVATION_H_ */
