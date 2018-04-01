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
#include "Utils.h"

template<class SavedStateActivation, class OutputActivation>
class RecurrentNeuralNetwork {
public:
	RecurrentNeuralNetwork(int x_size, int out_size, int saved_state_size);

	void feedForward(const Sentence& x, std::unique_ptr<arma::mat>& out_saved_states, std::unique_ptr<arma::mat>& out_outputs);

	const arma::mat& getW() const { return W; }
	const arma::mat& getU() const { return U; }
	const arma::mat& getV() const { return V; }
	void setW(const arma::mat& newW) { W = newW; }
	void setU(const arma::mat& newU) { U = newU; }
	void setV(const arma::mat& newV) { V = newV; }
private:
	arma::mat W;
	arma::mat U;
	arma::mat V;
};

#endif /* SRC_RECURRENTNEURALNETWORK_H_ */
