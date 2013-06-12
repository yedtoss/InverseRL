/*
 * FittedQIteration.h
 *
 *  Created on: 25 Sep 2012
 *      Author: yedtoss
 */

#ifndef FITTEDQITERATION_H_
#define FITTEDQITERATION_H_


#include "FittedQEval.h"
#include "ApproximatePolicyIteration.h"

class FittedQIteration : public ApproximatePolicyIteration {
public:

	FittedQEval policy_eval;
	FittedQIteration(double threshold_ =1e-90 , int num_iterations_ = 900){
		init(threshold_, num_iterations_);
	}

	void init(double& threshold_ , int& num_iterations_) {

			threshold = threshold_;
			num_iterations = num_iterations_;
		}



	virtual void initialize(VectorXd& w0, bool init = true){
			if(init){
				//pi = VectorXd::Zero(n_features);
				pi = VectorXd::Random(n_features);
			}
			else {
				pi = w0;
			}

		}



	virtual void  policy_evaluation(VectorXd& w0, AppEnvironment& env, Demonstration& D,
					ApproximateMDP& mdp) {

			policy_eval.compute(w0, env, D, mdp, &pi);
			epsilon = (pi-w0).squaredNorm();
			pi = w0;
			//std::cout<<"Diff pi "<<epsilon<<std::endl;
	}

	virtual ~FittedQIteration(){
		;
	}
};

#endif /* FITTEDQITERATION_H_ */
