/*
 * Main.cpp
 *
 *  Created on: Mar 28, 2018
 *      Author: danielruskin
 */
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

#include "TextVocabGenerator.h"
#include "TextRnn.h"
#include "TextActivationLossConfig.h"
#include "TextProgressEvaluator.h"
#include "NetworkTrainer.h"
#include "GradientChecker.h"

int main(int argc, char **argv)
{
	BOOST_LOG_TRIVIAL(info) << "Initializing...";

	boost::log::add_common_attributes();
	srand(time(0));
	arma::arma_rng::set_seed_random();

	BOOST_LOG_TRIVIAL(info) << "Parsing options...";
	boost::program_options::options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
	    ("data_file", boost::program_options::value<std::string>(), "A data file with text to use for training the model.")
		("text_vocab_size", boost::program_options::value<int>(), "The number of words N to include in the model.  Only the most recurring N words will be included; the rest will be set to an unknown value token.")
		("saved_state_size", boost::program_options::value<int>(), "The size of the saved state column vector.")
		("learning_rate", boost::program_options::value<double>(), "The learning rate for the model.")
		("num_epochs", boost::program_options::value<int>(), "Number of SGD epochs to run.")
		("num_samples_per_batch", boost::program_options::value<int>(), "Number of samples for each SGD minibatch.")
		("test_data_frac", boost::program_options::value<double>(), "Percent of data to use as test data (represented as decimal).")
		("bptt_truncate", boost::program_options::value<int>(), "Max number of BPTT steps to run when calculating the gradient for each example (-1 to not truncate).")
		("grad_check_h", boost::program_options::value<double>(), "If performing a grad check, the differential value to use.")
		("grad_check_error_threshold_pct", boost::program_options::value<double>(), "If performing a grad check, the percent error threshold (error must be over pct AND abs to count).")
		("grad_check_error_threshold_abs", boost::program_options::value<double>(), "If performing a grad check, the absolute error threshold (error must be over pct AND abs to count).")
		("predict_prev_output_folder", boost::program_options::value<std::string>(), ("Output folder where the model is saved, if you are loading an existing model for predictions."))
		("predict_text_gen_first_char", boost::program_options::value<std::string>(), ("The first character for text generation."))
		("predict_text_gen_last_char", boost::program_options::value<std::string>(), ("The first character for text generation."))
		("predict_text_gen_max_chars", boost::program_options::value<int>(), ("Max chars for text generation."))
		("command", boost::program_options::value<std::string>(), "The task to perform.  Must be one of: train_model, grad_check, predict_text_gen.")
		("output_prefix", boost::program_options::value<std::string>(), "Prefix for output files.  Can be a folder.")
	;
	boost::program_options::variables_map vm;
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
	boost::program_options::notify(vm);
	if (vm.count("help")) {
	    std::cout << desc << std::endl;
	    return 1;
	}

	BOOST_LOG_TRIVIAL(info) << "Validating options...";
	std::string cmd;
	if(vm.count("command")) {
		cmd = vm["command"].as<std::string>();
		std::vector<std::string> required_opts;

		if(cmd == "train_model") {
			required_opts = {
				"data_file",
				"text_vocab_size",
				"saved_state_size",
				"learning_rate",
				"num_epochs",
				"num_samples_per_batch",
				"test_data_frac",
				"bptt_truncate"
			};
		} else if(cmd == "grad_check") {
			required_opts = {
				"data_file",
				"text_vocab_size",
				"saved_state_size",
				"bptt_truncate",
				"grad_check_h",
				"grad_check_error_threshold_pct",
				"grad_check_error_threshold_abs",
			};
		} else if(cmd == "predict_text_gen") {
			required_opts = {
				"predict_prev_output_folder",
				"predict_text_gen_first_char",
				"predict_text_gen_last_char",
				"predict_text_gen_max_chars"
			};
		} else {
			throw std::runtime_error("An invalid command was specified!");
		}

		for(std::vector<std::string>::const_iterator it = required_opts.begin(); it != required_opts.end(); it++) {
			if(!vm.count(*it)) {
				throw std::runtime_error("Missing option: " + *it + "!");
			}
		}
	} else {
		throw std::runtime_error("No command was specified!");
	}

	BOOST_LOG_TRIVIAL(info) << "Executing command...";
	if(cmd == "train_model" || cmd == "grad_check") {
		std::string outputDir;
		if(vm.count("output_prefix")) {
			outputDir = vm["output_prefix"].as<std::string>();
		} else {
			outputDir = "./output-";
		}
		outputDir += boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time()) + "/";
		boost::filesystem::create_directories(outputDir);
		BOOST_LOG_TRIVIAL(info) << "Created output directory: " << outputDir;

		BOOST_LOG_TRIVIAL(info) << "Loading and parsing data...";
		std::ifstream data_file;
		data_file.open(vm["data_file"].as<std::string>());
		TextSentenceList data_loaded;
		std::string _tmp_line;
		while(getline(data_file, _tmp_line)) {
			TextSentence words;
			boost::split(words, _tmp_line, boost::is_any_of(" "));
			data_loaded.push_back(words);
		}

		BOOST_LOG_TRIVIAL(info) << "Creating vocab...";
		std::shared_ptr<TextVocab> text_vocab = std::move(generateVocab(vm["text_vocab_size"].as<int>(), data_loaded));

		BOOST_LOG_TRIVIAL(info) << "Creating network...";
		std::shared_ptr<TextRnn<TextActivationLossConfig>> network(new TextRnn<TextActivationLossConfig>(
			text_vocab->size(),
			text_vocab->size(),
			vm["saved_state_size"].as<int>(),
			text_vocab
		));

		BOOST_LOG_TRIVIAL(info) << "Converting training data to one-hot format...";
		std::unique_ptr<SentenceList> data_one_hot = network->textSentenceListToSentenceList(data_loaded);
		std::unique_ptr<SentenceList> data_final_x;
		std::unique_ptr<SentenceList> data_final_y;
		network->sentenceListToTrainingSentenceList(*data_one_hot, data_final_x, data_final_y);

		if(cmd == "train_model") {
			BOOST_LOG_TRIVIAL(info) << "Training network...";

			NetworkTrainer<TextRnn<TextActivationLossConfig>, TextActivationLossConfig, TextProgressEvaluator<TextActivationLossConfig>> trainer(
				vm["num_epochs"].as<int>(),
				vm["num_samples_per_batch"].as<int>(),
				vm["learning_rate"].as<double>(),
				vm["test_data_frac"].as<double>(),
				vm["bptt_truncate"].as<int>(),
				network
			);
			trainer.train(*data_final_x, *data_final_y);

			BOOST_LOG_TRIVIAL(info) << "Saving model...";
			network->save(outputDir);
		} else {
			BOOST_LOG_TRIVIAL(info) << "Saving model...";
			network->save(outputDir);

			BOOST_LOG_TRIVIAL(info) << "Saving first training example...";
			(*data_final_x)[0].save(outputDir + "example_x.csv", arma::csv_ascii);
			(*data_final_y)[0].save(outputDir + "example_y.csv", arma::csv_ascii);

			BOOST_LOG_TRIVIAL(info) << "Checking gradients with first example...";
			GradientChecker<TextRnn<TextActivationLossConfig>, TextActivationLossConfig> gradient_checker(
				vm["grad_check_h"].as<double>(),
				vm["grad_check_error_threshold_pct"].as<double>(),
				vm["grad_check_error_threshold_abs"].as<double>()
			);
			gradient_checker.checkGradients(outputDir, vm["bptt_truncate"].as<int>(), *network, (*data_final_x)[0], (*data_final_y)[0]);
		}
	} else if(cmd == "predict_text_gen") {
		BOOST_LOG_TRIVIAL(info) << "Loading model...";
		TextRnn<TextActivationLossConfig> model(vm["predict_prev_output_folder"].as<std::string>());

		BOOST_LOG_TRIVIAL(info) << "Generating text...";
		std::unique_ptr<TextSentence> sent = model.generateSentence(vm["predict_text_gen_first_char"].as<std::string>(), vm["predict_text_gen_last_char"].as<std::string>(), vm["predict_text_gen_max_chars"].as<int>());

		BOOST_LOG_TRIVIAL(info) << "Generated text!  Output: ";
		for(TextSentence::const_iterator it = sent->begin(); it != sent->end(); it++) {
			BOOST_LOG_TRIVIAL(info) << *it;
		}
	} else {
		throw std::runtime_error("An invalid command was specified!");
	}

	BOOST_LOG_TRIVIAL(info) << "Done!";
}
