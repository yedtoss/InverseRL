/*
 * LSPI.h
 *
 *  Created on: 8 Sep 2012
 *      Author: yedtoss
 */

#ifndef LSPI_H_
#define LSPI_H_


#include "ApproximatePolicyIteration.h"
#include "LSTDQ.h"

class LSPI : public ApproximatePolicyIteration {
public:
	LSTDQ policy_eval;
	LSPI(int reg = 2,double threshold_ =1e-90 , int num_iterations_ = 20){

		init(threshold_, num_iterations_);
		policy_eval = LSTDQ(2);

	}

	void init(double& threshold_ , int& num_iterations_) {

		threshold = threshold_;
		num_iterations = num_iterations_;
		
	}

	virtual void initialize(VectorXd& w0, bool init = true){
		if(init){
			pi = VectorXd::Zero(n_features);
		}
		else {
			pi = w0;
			//pi = VectorXd::Random(n_features);
		}

	}

	/// We don't need policy_improvement since LSTDQ is already doing exact improvement

	virtual void  policy_evaluation(VectorXd& w0, AppEnvironment& env, Demonstration& D,
				ApproximateMDP& mdp) {

		policy_eval.compute(w0, env, D, mdp, &pi);
		epsilon = (pi-w0).squaredNorm();
		pi = w0;
	}


	virtual ~LSPI(){

	}
};

#endif /* LSPI_H_ */
