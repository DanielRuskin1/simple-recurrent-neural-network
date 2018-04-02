/*
 * TextProgressEvaluator.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_TEXTPROGRESSEVALUATOR_H_
#define SRC_TEXTPROGRESSEVALUATOR_H_

#include "Utils.h"

template<class NetworkType>
class TextProgressEvaluator {
public:
	template<class NetworkType>
	static double evalPercentWordsCorrect(const NetworkType& network, const SentenceList& predict, const SentenceList& correct) {
		double correctAmount = 0;
		double totalAmount = 0;

		for(int sentence = 0; sentence < predict.size(); sentence++) {
			for(int word = 0; word < predict[sentence].n_cols; word++) {
				++totalAmount;
				if(*(network.wordToTextWord(predict[sentence].col(word))) == *(network.wordToTextWord(correct[sentence].col(word)))) {
					++correctAmount;
				}
			}
		}

		return correctAmount / totalAmount;
	}
};

#endif /* SRC_TEXTPROGRESSEVALUATOR_H_ */
