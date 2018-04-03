/*
 * Utils.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */
#include <unordered_map>
#include <armadillo>

// Word, sentence, etc. are defined in the "generic" sense.
// A word just represents a single prediction or example, it may not represent text (it could represent an image, etc).
typedef arma::colvec Word;
typedef arma::mat Sentence; // Colvec = word
typedef std::vector<Sentence> SentenceList; // Slice = sentence

typedef std::string TextWord;
typedef std::vector<TextWord> TextSentence;
typedef std::vector<TextSentence> TextSentenceList;

typedef std::unordered_map<TextWord, int> TextVocab;
typedef std::unordered_map<int, TextWord> TextVocabRev;
typedef std::vector<std::pair<TextWord, int>> TextOccurrenceCountsVec;
typedef std::unordered_map<TextWord, int> TextOccurrenceCountsMap;

#define UNKNOWN_CHAR_VAL "UNKNOWN_CHAR"
