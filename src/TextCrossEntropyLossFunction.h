/*
 * TextCrossEntropyLossFunction.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_TEXTCROSSENTROPYLOSSFUNCTION_H_
#define SRC_TEXTCROSSENTROPYLOSSFUNCTION_H_

// Gives the loss for a single word.  Sum to get total loss.
// SUM for all one-hot vec entries ( correct * log(predict) )
// For simplicity in gradient calculations later, we do not divide by N here.
double eval(const Word& predict, const Word& correct) {
	return arma::accu(-1 * correct % arma::log(predict));
}

// Gives the loss deriv w/r/t a single word.
// Derivative w/r/t a given one-hot-vec-entry is just -correct/predict for that entry.
std::unique_ptr<arma::colvec> evalPrime(const Sentence& predict, const Sentence& correct) {
	return std::unique_ptr<arma::colvec>(new arma::colvec(-1 * (correct / predict)));
}

#endif /* SRC_TEXTCROSSENTROPYLOSSFUNCTION_H_ */
