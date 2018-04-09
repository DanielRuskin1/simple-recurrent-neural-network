/*
 * GradientChecker.h
 *
 *  Created on: Apr 8, 2018
 *      Author: danielruskin
 */

#ifndef SRC_GRADIENTCHECKER_H_
#define SRC_GRADIENTCHECKER_H_

#include <boost/log/trivial.hpp>

template<class NetworkType, class ActivationLossConfig>
class GradientChecker {
public:
	GradientChecker(double h_new, double error_threshold_percent_new, double error_threshold_abs_new)
		: h(h_new), error_threshold_percent(error_threshold_percent_new), error_threshold_abs(error_threshold_abs_new) {

	}

	void checkGradients(const std::string& output_prefix, int bptt_truncate, const NetworkType& network, const Sentence& x, const Sentence& y) const {
		// Begin by performing feedforward/BPTT to get the calculated gradients for this example
		BOOST_LOG_TRIVIAL(info) << "Calculating gradients with BPTT...";
		std::unique_ptr<arma::mat> ff_saved_states;
		std::unique_ptr<Sentence> ff_outputs;
		std::unique_ptr<Sentence> bptt_dCdW;
		std::unique_ptr<Sentence> bptt_dCdU;
		std::unique_ptr<Sentence> bptt_dCdV;
		network.feedForward(x, ff_saved_states, ff_outputs);
		ActivationLossConfig::setGradients(network, bptt_truncate, x, y, *ff_saved_states, *ff_outputs, bptt_dCdW, bptt_dCdU, bptt_dCdV);

		// Copy the network to make changes to W/U/V
		NetworkType network_new = network;

		// Now, estimate each gradient for W/U/V
		BOOST_LOG_TRIVIAL(info) << "Estimating gradients with derivative method...";
		arma::mat est_dCdW(network.getW().n_rows, network.getW().n_cols, arma::fill::zeros);
		for(int row = 0; row < network.getW().n_rows; row++) {
			for(int col = 0; col < network.getW().n_cols; col++) {
				est_dCdW(row, col) = estimateGradient(network_new, "W", row, col, x, y);
			}
		}
		arma::mat est_dCdU(network.getU().n_rows, network.getU().n_cols, arma::fill::zeros);
		for(int row = 0; row < network.getU().n_rows; row++) {
			for(int col = 0; col < network.getU().n_cols; col++) {
				est_dCdU(row, col) = estimateGradient(network_new, "U", row, col, x, y);
			}
		}
		arma::mat est_dCdV(network.getV().n_rows, network.getV().n_cols, arma::fill::zeros);
		for(int row = 0; row < network.getV().n_rows; row++) {
			for(int col = 0; col < network.getV().n_cols; col++) {
				est_dCdV(row, col) = estimateGradient(network_new, "V", row, col, x, y);
			}
		}

		// Check if any failed
		BOOST_LOG_TRIVIAL(info) << "Evaluating gradients...";
		evalGradients(*bptt_dCdW, est_dCdW, "W");
		evalGradients(*bptt_dCdU, est_dCdU, "U");
		evalGradients(*bptt_dCdV, est_dCdV, "V");

		// Save BPTT/estimated gradients to file
		BOOST_LOG_TRIVIAL(info) << "Saving gradients...";
		boost::filesystem::create_directories(output_prefix + "gradients/");
		bptt_dCdW->save(output_prefix + "gradients/bptt_dCdW.csv", arma::csv_ascii);
		est_dCdW.save(output_prefix + "gradients/est_dCdW.csv", arma::csv_ascii);
		bptt_dCdU->save(output_prefix + "gradients/bptt_dCdU.csv", arma::csv_ascii);
		est_dCdU.save(output_prefix + "gradients/est_dCdU.csv", arma::csv_ascii);
		bptt_dCdV->save(output_prefix + "gradients/bptt_dCdV.csv", arma::csv_ascii);
		est_dCdV.save(output_prefix + "gradients/est_dCdV.csv", arma::csv_ascii);
	}
private:
	double h;
	double error_threshold_percent;
	double error_threshold_abs;

	double estimateGradient(NetworkType& network, const std::string& var, int row, int col, const Sentence& x, const Sentence& y) const {
		// Get cost for +H
		if(var == "W") {
			network.updateWVal(row, col, h);
		} else if(var == "U") {
			network.updateUVal(row, col, h);
		} else if(var == "V") {
			network.updateVVal(row, col, h);
		} else {
			throw std::runtime_error("Invalid param name!");
		}
		std::unique_ptr<arma::mat> ff_saved_states;
		std::unique_ptr<Sentence> ff_outputs;
		network.feedForward(x, ff_saved_states, ff_outputs);
		double costP = ActivationLossConfig::evalCost(y, *ff_outputs);

		// Get cost for -H
		if(var == "W") {
			network.updateWVal(row, col, -2*h);
		} else if(var == "U") {
			network.updateUVal(row, col, -2*h);
		} else if(var == "V") {
			network.updateVVal(row, col, -2*h);
		} else {
			throw std::runtime_error("Invalid param name!");
		}
		network.feedForward(x, ff_saved_states, ff_outputs);
		double costN = ActivationLossConfig::evalCost(y, *ff_outputs);

		// Return network to original value
		if(var == "W") {
			network.updateWVal(row, col, h);
		} else if(var == "U") {
			network.updateUVal(row, col, h);
		} else if(var == "V") {
			network.updateVVal(row, col, h);
		} else {
			throw std::runtime_error("Invalid param name!");
		}

		// Return calculated gradient
		return (costP - costN) / (2.0 * h);
	}

	void evalGradients(const arma::mat& bptt, const arma::mat& est, const std::string& var) const {
		for(int row = 0; row < bptt.n_rows; row++) {
			for(int col = 0; col < bptt.n_cols; col++) {
				double bptt_v = bptt(row, col);
				double est_v = est(row, col);

				if(std::abs(bptt_v - est_v) / std::abs(bptt_v + est_v) > error_threshold_percent && std::abs(bptt_v - est_v) > error_threshold_abs) {
					BOOST_LOG_TRIVIAL(info) << "Gradient check ERROR: var=" << var << " row=" << row << " col=" << col << " BPTT=" << bptt_v << " EST=" << est_v;
				}
			}
		}
	}
};

#endif /* SRC_GRADIENTCHECKER_H_ */
