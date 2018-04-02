/*
 * TextRnn.cpp
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#include "TextRnn.h"

template<class ActivationLossConfig>
TextRnn<ActivationLossConfig>::TextRnn(int x_size, int out_size, int saved_state_size, const std::shared_ptr<TextVocab> new_vocab)
	: RecurrentNeuralNetwork<ActivationLossConfig>(x_size, out_size, saved_state_size) {
	vocab = new_vocab;

	for(std::unordered_map<TextWord, int>::const_iterator it = vocab->begin(); it != vocab->end(); it++) {
		vocab_rev[it->second] = it->first;
	}
}

template<class ActivationLossConfig>
std::unique_ptr<Word> TextRnn<ActivationLossConfig>::textWordToWord(const TextWord& text_word) const {
	std::unique_ptr<Word> ret(new Word(vocab->size(), arma::fill::zeros));
	(*ret)(vocab->at(text_word)) = 1;

	return ret;
}

template<class ActivationLossConfig>
std::unique_ptr<TextWord> TextRnn<ActivationLossConfig>::wordToTextWord(const Word& word) const {
	return std::unique_ptr<Word>(new TextWord(vocab_rev.at(word.index_max())));
}

template<class ActivationLossConfig>
std::unique_ptr<Sentence> TextRnn<ActivationLossConfig>::textSentenceToSentence(const TextSentence& text_sentence) const {
	std::unique_ptr<Sentence> ret(new Sentence(vocab->size(), text_sentence.size(), arma::fill::zeros));

	for(int at = 0; at < text_sentence.size(); at ++) {
		(*ret)(at, vocab->at(text_sentence[at])) = 1;
	}

	return ret;
}

template<class ActivationLossConfig>
std::unique_ptr<TextSentence> TextRnn<ActivationLossConfig>::sentenceToTextSentence(const Sentence& sentence) const {
	std::unique_ptr<TextSentence> ret(new TextSentence);

	for(int at = 0; at < sentence.n_cols(); at ++) {
		ret->push_back(vocab_rev.at(sentence.col(at).index_max()));
	}

	return ret;
}

template<class ActivationLossConfig>
std::unique_ptr<SentenceList> TextRnn<ActivationLossConfig>::textSentenceListToSentenceList(const TextSentenceList& text_sentences) const {
	std::unique_ptr<SentenceList> ret(new SentenceList);

	for(int at = 0; at < text_sentences.size(); at++) {
		ret->push_back(std::move(*(textSentenceToSentence(text_sentences[at])))); // TODO: is this correct?  If so, use more.
	}

	return ret;
}

template<class ActivationLossConfig>
std::unique_ptr<TextSentenceList> TextRnn<ActivationLossConfig>::sentenceListToTextSentenceList(const SentenceList& sentences) const {
	std::unique_ptr<TextSentenceList> ret(new TextSentenceList);

	for(int at = 0; at < sentences.size(); at++) {
		ret->push_back(std::move(*(sentenceToTextSentence(sentences[at]))));
	}

	return ret;
}

template<class ActivationLossConfig>
std::unique_ptr<TextSentence> TextRnn<ActivationLossConfig>::generateSentence(const TextWord& word_zero, const TextWord& end_token, int max_words) const {
	std::unique_ptr<TextSentence> ts(new TextSentence);
	ts->push_back(word_zero);

	arma::colvec last_saved_state(this->W.n_rows);
	Word last_output = *(textWordToWord(word_zero));
	while(!(ts->back() == end_token || ts->size() < max_words)) {
		last_saved_state = *(ActivationLossConfig::evalSavedStateActivation(
			(this->U * last_output) + (this->W * last_saved_state)
		));
		last_output = *(ActivationLossConfig::evalOutputActivation(this->V * last_saved_state));
		ts->push_back(*(wordToTextWord(last_output)));
	}

	return ts;
}
