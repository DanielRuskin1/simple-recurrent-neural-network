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

template<class ActivationLossConfig>
class RecurrentNeuralNetwork {
public:
	RecurrentNeuralNetwork(int x_size, int out_size, int saved_state_size) {
		// Fill to +-[0.01, 1.01] and divide by 10 => [0.001, 0.101]
		W.resize(saved_state_size, saved_state_size);
		W.fill(arma::fill::randn);
		W /= 10;
		U.resize(saved_state_size, x_size);
		U.fill(arma::fill::randn);
		U /= 10;
		V.resize(out_size, saved_state_size);
		V.fill(arma::fill::randn);
		V /= 10;
	}

	// In input, each col is one X value.
	// In return vals, each col is one saved state/output.
	// First saved state is always 0.
	void feedForward(const Sentence& x, std::unique_ptr<arma::mat>& out_saved_states, std::unique_ptr<arma::mat>& out_outputs) {
		out_saved_states.reset(new arma::mat(W.n_rows, x.n_cols + 1, arma::fill::zeros));
		out_outputs.reset(new arma::mat(V.n_rows, x.n_cols, arma::fill::zeros));

		// Iterate through each X value and calculate/save the new saved state/output
		for(int x_iter = 0; x_iter < x.n_cols; x_iter++) {
			out_saved_states->col(x_iter + 1) = *(ActivationLossConfig::evalSavedStateActivation(
				(U * x.col(x_iter)) + (W * out_saved_states->col(x_iter))
			));
			out_outputs->col(x_iter) = *(ActivationLossConfig::evalOutputActivation(V * out_saved_states->col(x_iter + 1)));
		}
	}

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
