/*
 * TextCrossEntropyLossFunction.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_TEXTCROSSENTROPYLOSSFUNCTION_H_
#define SRC_TEXTCROSSENTROPYLOSSFUNCTION_H_

// SUM for all one-hot vec entries ( correct * log(predict) )
// For simplicity in gradient calculations later, we do not divide by N here.
double eval(Word predict, Word correct) {
	return arma::accu(-1 * correct % arma::log(predict));
}

// Derivative w/r/t a given one-hot-vec-entry is just -correct/predict for that entry.
arma::colvec evalPrime(Sentence predict, Sentence correct) {
	return -1 * (correct / predict);
}

#endif /* SRC_TEXTCROSSENTROPYLOSSFUNCTION_H_ */
