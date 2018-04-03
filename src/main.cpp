/*
 * Main.cpp
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */
#include <iostream>
#include <fstream>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>

#include "TextVocabGenerator.h"
#include "TextRnn.h"
#include "TextActivationLossConfig.h"
#include "TextProgressEvaluator.h"
#include "NetworkTrainer.h"

int main(int argc, char **argv)
{
	BOOST_LOG_TRIVIAL(info) << "Initializing...";

	// Setup logging/random
	boost::log::add_common_attributes();
	srand(time(0));
	arma::arma_rng::set_seed_random();

	BOOST_LOG_TRIVIAL(info) << "Loading data file...";

	// Load data file
	std::ifstream data_file;
	std::string line;
	TextSentenceList data;
	data_file.open("../data/parsed.txt");
	while(getline(data_file, line)) {
		TextSentence words;
		boost::split(words, line, boost::is_any_of(" "));
		data.push_back(words);
	}

	BOOST_LOG_TRIVIAL(info) << "Creating vocab...";

	// Create vocab with top 5000 words
	std::shared_ptr<TextVocab> tv = std::move(generateVocab(5000, data));

	BOOST_LOG_TRIVIAL(info) << "Creating network...";

	// Create text network
	std::shared_ptr<TextRnn<TextActivationLossConfig>> trnn(new TextRnn<TextActivationLossConfig>(
		5001,
		5001,
		100,
		tv
	));

	BOOST_LOG_TRIVIAL(info) << "Processing training data...";

	// Convert our data to valid training data
	std::unique_ptr<SentenceList> data_one_hot = trnn->textSentenceListToSentenceList(data);
	SentenceList::const_iterator it_a = data_one_hot->begin();
	SentenceList::const_iterator it_b = data_one_hot->end() - 1;
	SentenceList x(it_a, it_b - 1);
	SentenceList y(it_a + 1, it_b);

	BOOST_LOG_TRIVIAL(info) << "Training network...";
	NetworkTrainer<TextRnn<TextActivationLossConfig>, TextActivationLossConfig, TextProgressEvaluator<TextActivationLossConfig>> nnt(
		10,
		100,
		0.1,
		0.05,
		5,
		trnn
	);
	nnt.train(x, y);
}
