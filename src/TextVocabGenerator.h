/*
 * TextVocabGenerator.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_TEXTVOCABGENERATOR_H_
#define SRC_TEXTVOCABGENERATOR_H_

#include <unordered_map>
#include "Utils.h"

std::unique_ptr<TextVocab> generateVocab(int max_vocab, const TextSentenceList& sentences) {
	// Get occurrence counts for each word (roughly O(n)
	TextOccurrenceCountsMap occurrence_counts;
	for(int sentence = 0; sentence < sentences.size(); sentence++) {
		for(TextSentence::const_iterator i = sentences[sentence].begin(); i != sentences[sentence].end(); i++) {
			if(occurrence_counts.find(*i) == occurrence_counts.end()) {
				occurrence_counts[*i] = 1;
			} else {
				occurrence_counts[*i] += 1;
			}
		}
	}

	// Get top n occurring words
	TextOccurrenceCountsVec occurrence_counts_pairs(occurrence_counts.begin(), occurrence_counts.end());
	std::nth_element(
		occurrence_counts_pairs.begin(),
		occurrence_counts_pairs.begin() + max_vocab - 1,
		occurrence_counts_pairs.end()
	);

	std::unique_ptr<TextVocab> res(new TextVocab(occurrence_counts_pairs.begin(), occurrence_counts_pairs.begin() + max_vocab));
	(*res)[UNKNOWN_CHAR_VAL] = max_vocab;

	return res;
}

#endif /* SRC_TEXTVOCABGENERATOR_H_ */
