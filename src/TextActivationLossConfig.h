/*
 * TextActivationLossConfig.h
 *
 *  Created on: Apr 2, 2018
 *      Author: danielruskin
 */

#ifndef SRC_TEXTACTIVATIONLOSSCONFIG_H_
#define SRC_TEXTACTIVATIONLOSSCONFIG_H_

#include <memory>
#include <armadillo>
#include "RecurrentNeuralNetwork.h"
#include "TextRnn.h"

// Output activation: softmax
// Saved state activation: tanh
// Cost Function: cross-entropy loss
class TextActivationLossConfig {
public:
	static std::unique_ptr<arma::colvec> evalOutputActivation(const arma::colvec& in);
	static std::unique_ptr<arma::colvec> evalSavedStateActivation(const arma::colvec& in);

	static void addGradients(const TextRnn<TextActivationLossConfig>& network, int bptt_truncate, const Sentence& x, const Sentence& y, const arma::mat& saved_states, const arma::mat& outputs, arma::mat& out_dCdW, arma::mat& out_dCdU, arma::mat& out_dCdV);
};

#endif /* SRC_TEXTACTIVATIONLOSSCONFIG_H_ */
