/*
 * TextRnn.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_TEXTRNN_H_
#define SRC_TEXTRNN_H_

#include "Utils.h"

template<class SavedStateActivation, class OutputActivation>
class TextRnn : public RecurrentNeuralNetwork<SavedStateActivation, OutputActivation> {
public:
	TextRnn(const std::shared_ptr<TextVocab> new_vocab);

	std::unique_ptr<Word> textWordToWord(const TextWord& text_word) const; // Convert to one-hot
	std::unique_ptr<TextWord> wordToTextWord(const Word& word) const; // Convert from one-hot
	std::unique_ptr<Sentence> textSentenceToSentence(const TextSentence& text_sentence) const; // Convert to one-hot
	std::unique_ptr<TextSentence> sentenceToTextSentence(const Sentence& sentence) const; // Convert from one-hot
	std::unique_ptr<SentenceList> textSentenceListToSentenceList(const TextSentenceList& text_sentences) const; // Convert to one-hot
	std::unique_ptr<TextSentenceList> sentenceListToTextSentenceList(const SentenceList& sentences) const; // Convert from one-hot

	std::unique_ptr<TextSentence> generateSentence(const TextWord& word_zero) const;
private:
	const std::shared_ptr<TextVocab> vocab;
};

#endif /* SRC_TEXTRNN_H_ */
