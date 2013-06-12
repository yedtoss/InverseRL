/*
 * ApproximatePolicyIteration.h
 *
 *  Created on: 7 Sep 2012
 *      Author: yedtoss
 */

#ifndef APPROXIMATEPOLICYITERATION_H_
#define APPROXIMATEPOLICYITERATION_H_

#include <Eigen/Dense>
#include <iostream>


#include "MyTypes.h"

class ApproximatePolicyIteration {
public:

	int n_episodes, n_samples, n_features, n_states, n_actions, num_iterations, epsilon;
	double threshold;
	VectorXd w, pi;
	int tailledemo;
	//ApproximatePolicyIteration(){

	//}

	virtual void compute(VectorXd& w0, AppEnvironment& env, Demonstration& D,
			ApproximateMDP& mdp, bool init = true){

		n_features = env.get_q_num_training();
		n_states = env.get_n_states();
		n_actions = env.get_n_actions();
		assert(n_features > 0);
		assert(n_states > 0);
		int iter = 0;

		initialize(w0);

		tailledemo = D.num_demo;



		epsilon = 3000;

		while(epsilon>threshold && iter < num_iterations){
			policy_evaluation(w0, env, D, mdp);
			policy_improvement(w0, env, D, mdp);
			//std::cout<<"it "<<iter<<std::endl;
			iter++;
			//epsilon++;
			//std::cout<<"Iter "<<iter<<std::endl;
			//std::cout<<"Epsilon "<<epsilon<<" "<<iter<<std::endl;

		}
		assert(iter>0);
		finalize(w0);

	}

	virtual void  policy_evaluation(VectorXd& w0, AppEnvironment& env, Demonstration& D,
			ApproximateMDP& mdp) {

	}
	virtual void  policy_improvement(VectorXd& w0, AppEnvironment& env, Demonstration& D,
			ApproximateMDP& mdp) {

	}
	virtual void initialize(VectorXd& w0, bool init = true){

	}


	virtual void finalize(VectorXd& w0){

	}
	virtual ~ApproximatePolicyIteration(){

	}
};

#endif /* APPROXIMATEPOLICYITERATION_H_ */
