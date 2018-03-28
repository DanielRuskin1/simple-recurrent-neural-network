/*
 * Utils.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */
#include <map>

// Word, sentence, etc. are defined in the "generic" sense.
// A word just represents a single prediction or example, it may not represent text (it could represent an image, etc).
typedef arma::colvec Word;
typedef arma::mat Sentence; // Colvec = word
typedef std::vector<arma::mat> SentenceList; // Slice = sentence

typedef std::string TextWord;
typedef std::vector<TextWord> TextSentence;
typedef std::vector<TextSentence> TextSentenceList;

typedef std::map<TextWord, int> TextVocab;
