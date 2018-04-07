/*
 * TextRnn.h
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */

#ifndef SRC_TEXTRNN_H_
#define SRC_TEXTRNN_H_

#include <memory>
#include "Utils.h"
#include "RecurrentNeuralNetwork.h"

// TODO: Does public inheritance mean that the superclass's ctor will also be available here?
//		 If so, need to fix that.
template<class ActivationLossConfig>
class TextRnn : public RecurrentNeuralNetwork<ActivationLossConfig> {
public:
	TextRnn(int x_size, int out_size, int saved_state_size, std::shared_ptr<const TextVocab> new_vocab)
		: RecurrentNeuralNetwork<ActivationLossConfig>(x_size, out_size, saved_state_size) {
		vocab = new_vocab;

		for(std::unordered_map<TextWord, int>::const_iterator it = vocab->begin(); it != vocab->end(); it++) {
			vocab_rev[it->second] = it->first;
		}
	}

	std::unique_ptr<Word> textWordToWord(const TextWord& text_word) const {
		std::unique_ptr<Word> ret(new Word(vocab->size(), arma::fill::zeros));
		if(vocab->find(text_word) == vocab->end()) {
			(*ret)(vocab->at(UNKNOWN_CHAR_VAL)) = 1;
		} else {
			(*ret)(vocab->at(text_word)) = 1;
		}

		return ret;
	}

	std::unique_ptr<TextWord> wordToTextWord(const Word& word) const {
		return std::unique_ptr<TextWord>(new TextWord(vocab_rev.at(word.index_max())));
	}

	std::unique_ptr<Sentence> textSentenceToSentence(const TextSentence& text_sentence) const {
		std::unique_ptr<Sentence> ret(new Sentence(vocab->size(), text_sentence.size(), arma::fill::zeros));

		for(int at = 0; at < text_sentence.size(); at ++) {
			if(vocab->find(text_sentence[at]) == vocab->end()) {
				(*ret)(vocab->at(UNKNOWN_CHAR_VAL), at) = 1;
			} else {
				(*ret)(vocab->at(text_sentence[at]), at) = 1;
			}
		}

		return ret;
	}

	std::unique_ptr<TextSentence> sentenceToTextSentence(const Sentence& sentence) const {
		std::unique_ptr<TextSentence> ret(new TextSentence);

		for(int at = 0; at < sentence.n_cols(); at ++) {
			ret->push_back(vocab_rev.at(sentence.col(at).index_max()));
		}

		return ret;
	}

	std::unique_ptr<SentenceList> textSentenceListToSentenceList(const TextSentenceList& text_sentences) const {
		std::unique_ptr<SentenceList> ret(new SentenceList);

		for(int at = 0; at < text_sentences.size(); at++) {
			ret->push_back(std::move(*(textSentenceToSentence(text_sentences[at])))); // TODO: is this correct?  If so, use more.
		}

		return ret;
	}

	void sentenceListToTrainingSentenceList(const SentenceList& sl, std::unique_ptr<SentenceList>& out_x, std::unique_ptr<SentenceList>& out_y) {
		out_x.reset(new SentenceList);
		out_y.reset(new SentenceList);

		for(int ex = 0; ex < sl.size(); ex++) {
			out_x->push_back(sl[ex].cols(0, sl[ex].n_cols - 2)); // Drop last word for X
			out_y->push_back(sl[ex].cols(1, sl[ex].n_cols - 1)); // Drop first word for Y
		}
	}

	std::unique_ptr<TextSentenceList> sentenceListToTextSentenceList(const SentenceList& sentences) const {
		std::unique_ptr<TextSentenceList> ret(new TextSentenceList);

		for(int at = 0; at < sentences.size(); at++) {
			ret->push_back(std::move(*(sentenceToTextSentence(sentences[at]))));
		}

		return ret;
	}

	std::unique_ptr<TextSentence> generateSentence(const TextWord& word_zero, const TextWord& end_token, int max_words) const {
		std::unique_ptr<TextSentence> ts(new TextSentence);
		ts->push_back(word_zero);

		arma::colvec last_saved_state(this->W.n_rows);
		Word last_output = *(textWordToWord(word_zero));
		while(ts->back() != end_token && ts->size() < max_words) {
			last_saved_state = *(ActivationLossConfig::evalSavedStateActivation(
				(this->U * last_output) + (this->W * last_saved_state)
			));
			last_output = *(ActivationLossConfig::evalOutputActivation(this->V * last_saved_state));
			ts->push_back(*(wordToTextWord(last_output)));
		}

		return ts;
	}

private:
	std::shared_ptr<const TextVocab> vocab;
	TextVocabRev vocab_rev;
};

#endif /* SRC_TEXTRNN_H_ */
