/*
 * TDLearning.h
 *
 *  Created on: 7 Sep 2012
 *      Author: yedtoss
 */

#ifndef TDLEARNING_H_
#define TDLEARNING_H_

#include <iostream>
#include <cmath>



#include "AppEnvironment.h"
#include "MyTypes.h"


class TDLearning {
public:



	int n_episodes, n_samples, n_features, n_states, n_actions, s, s_next, a, nep, ns;
	double delta, gamma2;
	int tailledemo;
	int iter;
	bool fast;
	//VectorXd phi,phi_next;

	//TDLearning(){

	//}

	void compute(VectorXd& theta, AppEnvironment& env, Demonstration& D, ApproximateMDP& mdp,
			VectorXd* pi = NULL, VectorXd* b = NULL){

		n_features = env.get_q_num_training();
		n_states = env.get_n_states();
		n_actions = env.get_n_actions();
		assert(n_features > 0);
		assert(n_states > 0);
		iter = 0;

		tailledemo = D.num_demo;
		assert(tailledemo > 0);

		initialize(theta);

		if(!fast){

		for(nep = 0; nep < D.size(); nep++){

			for(ns = 0; ns < D[nep].size(); ns++){

				s = D[nep][ns].s;
				a = D[nep][ns].a;
				s_next = D[nep][ns].s_next;
				if(s_next <= -1){
					continue;
				}

				// Terminal state

				/*if(ns == D[nep].size() - 1){

					continue;
				}*/
				update(theta, env, D, mdp, pi, b);
				iter++;

			}
		}
		}
		else{

			iter =2;
		}

		finalize(theta, env, D, mdp);
		assert(iter>0);

		//for(int i = 0; i < n_features; i++){

			//assert(!std::isnan(theta(i)));
			//assert(!std::isinf(theta(i)));
		//}

	}

	virtual void static_settings(AppEnvironment& env, Demonstration& D){

	}

	virtual void initialize(VectorXd& theta){

	}

	virtual void finalize(VectorXd& theta,  AppEnvironment& env,  Demonstration& D, ApproximateMDP& mdp){

	}

	virtual void update(VectorXd& theta, AppEnvironment& env, Demonstration& D, ApproximateMDP& mdp,
				VectorXd* pi = NULL, VectorXd* b = NULL){
				
				//std::cout<<"m ";

	}
	virtual ~TDLearning()
	{

	}
};

#endif /* TDLEARNING_H_ */
