/*
 * Optimize.h
 *
 *  Created on: 9 Sep 2012
 *      Author: yedtoss
 */

#ifndef OPTIMIZE_H_
#define OPTIMIZE_H_


#include "IRL.h"
#include <nlopt.hpp>

std::vector<double> t;

double f(const std::vector<double> &x, std::vector<double>&grad, void*f_data) {

	std::vector<double>& y = const_cast<std::vector<double>& > (x);
	//std::vector<double> y = x;
	return ((IRL*) f_data)->likelihood(y,((IRL*) f_data)->mode);

	double tmp = 0;

	for (int i = 0; i < (int) x.size(); i++) {
		tmp += std::pow(x[i] - t[i], 2);

	}
	return tmp;
}

double f2(const std::vector<double> &x, std::vector<double>&grad, void*f_data){

	std::vector<double>& y = const_cast<std::vector<double>& > (x);
	//std::vector<double>& g = const_cast<std::vector<double>& > (grad);

	//double b = ((IRL*) f_data)->likelihood(y,((IRL*) f_data)->mode);
	double b = ((IRL*) f_data)->moi(y);

	std::cout<<" cur "<<b<<std::endl;

	grad = ((IRL*) f_data)->gradient2();

	return b;



}


void computeOpt(VectorXd& w_found, AppEnvironment& env,
		Demonstration& D, double max_time = 720, int n_iterations =
				200, int max_eval = 50000, double threshold = 0.000000000002,
		IRL_MODE mode = IRL_CRPB, Demonstration* Dlspi_=NULL) {
	IRL *myirl = new IRL(env, D,mode,Dlspi_);
	//myirl->Db = &D;
	//myirl->envb = &env;
	//myirl->modeb = mode;

	//if (mode == IRL_CJ || mode == IRL_CJA) {

		//myirl->computeFrequency(D);

	//}
	//myirl->n_features = env.get_n_features();
	if(mode == IRL_CPRB)
		myirl->n_features = env.get_q_num_training();
	else
		myirl->n_features = env.get_num_training();


	myirl->opt_policy = VectorXd::Zero(env.get_q_num_training());






	/*column_vector starting_point;
			starting_point.set_size(myirl->n_features);
			for (int i = 0; i < myirl->n_features; i++) {
				 //starting_point(i) = ((double) (rand() % 20)) / 23.0;
				 starting_point(i) = 0.00;
			}

			dlib::find_max(dlib::lbfgs_search_strategy(10),  // The 10 here is basically a measure of how much memory L-BFGS will use.
			                 dlib::objective_delta_stop_strategy(1e-7).be_verbose(),  // Adding be_verbose() causes a message to be
			                                                                    // printed for each iteration of optimization.
			                 ,
			                , starting_point, -1);

			myirl->moi(starting_point);*/















	std::vector<double> l(myirl->n_features, -3), u(myirl->n_features, 3), s(
			myirl->n_features, 0.5);
	t.resize(myirl->n_features, 0.5);
	double o = 0;

	//nlopt::opt z(nlopt::LN_COBYLA,myirl->n_features);
	nlopt::opt z(nlopt::GN_DIRECT_L, myirl->n_features);
	//nlopt::opt z(nlopt::LN_SBPLX,myirl->n_features);
	//nlopt::opt z(nlopt::GN_ISRES,myirl->n_features);
	//nlopt::opt z(nlopt::GN_CRS2_LM,myirl->n_features);
	//nlopt::opt z(nlopt::LN_BOBYQA,myirl->n_features);
	//nlopt::opt z(nlopt::LD_LBFGS,myirl->n_features);
	//nlopt::opt z(nlopt::LD_TNEWTON_PRECOND_RESTART,myirl->n_features);
	z.set_min_objective(f, myirl);
	//z.set_max_objective(f2, myirl);
	z.set_lower_bounds(l);
	z.set_upper_bounds(u);
	z.set_stopval(threshold);
	z.set_maxeval(max_eval);
	z.set_maxtime(max_time);
	//z.verbose = 1;
	//w_found.Resize(myirl->n_features, 1);
	w_found = Eigen::VectorXd::Zero(myirl->n_features);
	if (z.optimize(s, o) > 0) {
		std::cout << "The optimal value found is :\t";
		for (int i = 0; i < (int) s.size(); i++) {
			std::cout << s[i] << "\t";
			w_found(i) = s[i];
		}


		if(mode == IRL_CRPB){

			ApproximateMDP mdp;
			mdp.R = w_found;

			//myirl->lspi->compute(w_found, env, D, mdp);

		}



		std::cout << "The corresponding objective is :\t" << o << std::endl;
	}

	else {
		std::cout << "Error" << std::endl;
	}

	//myirl->likelihood(arg);

}



#endif /* OPTIMIZE_H_ */
