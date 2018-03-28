/*
 * RecurrentNeuralNetwork.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_RECURRENTNEURALNETWORK_H_
#define SRC_RECURRENTNEURALNETWORK_H_

#include <armadillo>
#include <memory>

template<class SavedStateActivation, class OutputActivation>
class RecurrentNeuralNetwork {
public:
	RecurrentNeuralNetwork();

	void feedForward(const arma::mat& x, std::unique_ptr<arma::mat>& out_saved_states, std::unique_ptr<arma::mat>& out_outputs);

	const arma::mat& getW() const { return W; }
	const arma::mat& getU() const { return U; }
	const arma::mat& getV() const { return V; }
private:
	arma::mat W;
	arma::mat U;
	arma::mat V;
};

#endif /* SRC_RECURRENTNEURALNETWORK_H_ */
