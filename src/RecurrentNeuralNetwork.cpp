/*
 * RecurrentNeuralNetwork.cpp
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#include "RecurrentNeuralNetwork.h"

template<class SavedStateActivation, class OutputActivation>
RecurrentNeuralNetwork<SavedStateActivation, OutputActivation>::RecurrentNeuralNetwork(int x_size, int out_size, int saved_state_size) {
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
template<class SavedStateActivation, class OutputActivation>
void RecurrentNeuralNetwork<SavedStateActivation, OutputActivation>::feedForward(const Sentence& x, std::unique_ptr<arma::mat>& out_saved_states, std::unique_ptr<arma::mat>& out_outputs) {
	out_saved_states.reset(new arma::mat(W.n_rows, x.n_cols + 1, arma::fill::zeros));
	out_outputs.reset(new arma::mat(V.n_rows, x.n_cols, arma::fill::zeros));

	// Iterate through each X value and calculate/save the new saved state/output
	for(int x_iter = 0; x_iter < x.n_cols; x_iter++) {
		out_saved_states->col(x_iter + 1) = *(SavedStateActivation::eval(
			(U * x.col(x_iter)) + (W * out_saved_states->col(x_iter))
		));
		out_outputs->col(x_iter) = *(OutputActivation::eval(V * out_saved_states->col(x_iter + 1)));
	}
}
