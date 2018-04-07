/*
 * TextActivationLossConfig.cpp
 *
 *  Created on: Apr 2, 2018
 *      Author: danielruskin
 */

#include "TextActivationLossConfig.h"

std::unique_ptr<arma::colvec> TextActivationLossConfig::evalOutputActivation(const arma::colvec& in) {
	arma::colvec matExp = arma::exp(in);

	return std::unique_ptr<arma::colvec>(new arma::colvec(matExp / arma::accu(matExp)));
}

std::unique_ptr<arma::colvec> TextActivationLossConfig::evalSavedStateActivation(const arma::colvec& in) {
	return std::unique_ptr<arma::colvec>(new arma::colvec(arma::tanh(in)));
}

// Everywhere we use saved_state, we increase the index by 1, because the first saved_state is for time step -1.
void TextActivationLossConfig::addGradients(const TextRnn<TextActivationLossConfig>& network, int bptt_truncate, const Sentence& x, const Sentence& y, const arma::mat& saved_states, const arma::mat& outputs, arma::mat& out_dCdW, arma::mat& out_dCdU, arma::mat& out_dCdV) {
	for(int time = x.n_cols - 1; time >= 0; time--) {
		// Derivative of cost for this time w/r/t V is trivial
		out_dCdV += arma::kron((outputs.col(time) - y.col(time)), arma::trans(saved_states.col(time + 1)));

		/*
		 * Calculate some common terms for dC/dW and dC/dU
		 */
		// Output of this is the derivative of cost w/r/t the Kth saved state
		arma::mat common_term_a = arma::trans(network.getV()) * (outputs.col(time) - y.col(time));

		// Output of this is the derivative of the Kth saved state w/r/t the Kth tanh input
		arma::mat common_term_b = 1 - arma::pow(saved_states.col(time + 1), 2);

		// Derivative of cost w/r/t the Kth tanh input.
		arma::mat common_term_c = common_term_a % common_term_b;

		for(int time_inner = time; time_inner >= std::max((time - bptt_truncate), 0); time_inner--) {
			// For the first iteration, this is just the
			// derivative of cost w/r/t weight for the current time_inner step.
			// (1,1) => (derivative of cost w/r/t 1st saved state) * (derivative of 1st saved state w/r/t W(1,1))
			// (1,2) => (derivative of cost w/r/t 1st saved state) * (derivative of 1st saved state w/r/t W(1,2))
			out_dCdW += arma::kron(common_term_c, arma::trans(saved_states.col(time_inner - 1 + 1)));

			// Derivative of cost w/r/t U is the same as w/r/t W for every term,
			// except the final factor is X instead of the previous saved state.
			out_dCdU += arma::kron(common_term_c, arma::trans(x.col(time_inner)));

			// For the second iteration, dCdW should add the second term:
			// => Derivative of dCdW for the time'th example, w/r/t the weights applied to the saved state for T-2
			// To get this for W(I, J), we can do the following (sum over K):
			// (1) Derivative of cost w/r/t Kth tan input (already set in common_term_c)
			// Times (2) Derivative of Kth tan input w/r/t Ith saved state T-1 (W(K, I))
			// Times (3) Derivative of Ith saved state T-2 w/r/t I tanh T-1 input (tanh' for T-1)
			// Times (4) Derivative of I tanh input T-1 w/r/t W_ij (S_j for T-2)
			// But if we just update common_term_c to be sum of 1*2*3 for all K,
			// it will already be multiplied by 4 in the next iteration.
			// So that's what we do here.
			// This is equivalent to setting common_term_c(K) to the derivative of the cost w/r/t the
			// Kth tanh input T-1.
			common_term_c = (arma::trans(network.getW()) * common_term_c) % (1 - arma::pow(saved_states.col(time - 1 + 1), 2));
		}
	}
}
