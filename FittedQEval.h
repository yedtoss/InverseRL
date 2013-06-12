/*
 * FittedQEval.h
 *
 *  Created on: 13 Sep 2012
 *      Author: yedtoss
 */

#ifndef FITTEDQEVAL_H_
#define FITTEDQEVAL_H_

#include "TDLearning.h"
#include "Random2.h"

class FittedQEval: public TDLearning {
public:

	int ind;
	VectorXd Q;
	vi Q_action;
	int max_time;
	int n_iterations;
	int freq;
	int* ind_j;
	int* sign_j;
	double outmin,outmax,A,B,netmin,netmax;
	FittedQEval() {
		ind = 0;
		max_time = 400;
		n_iterations = 60;
		//n_iterations = 20;
		//freq = 90;
		freq = 20;
		gamma2 = 1;
		A=1;
		B=0;
		fast = false;
	}



	virtual void initialize(VectorXd& theta) {

		ind = 0;
		Q = VectorXd::Zero(tailledemo);
		outmin = 5000;
		outmax = -5000;
		//Q_action = VectorXd::Zero(tailledemo);
		precompute();

	}

	virtual void static_settings(AppEnvironment& env, Demonstration& D){
		std::vector<VectorXd> input;

		int num_ins=0;

		for (nep = 0; nep < D.size(); nep++) {

					for (ns = 0; ns < D[nep].size(); ns++) {

						if (D[nep][ns].s_next <= -1) {
							continue;
						}

						num_ins++;

						input.push_back(env.get_basis(nep, ns, DEMO_ACT_NEXT, D[nep][ns].s,D[nep][ns].a));
						Q_action.push_back(D[nep][ns].a);
					}
		}

		env.qapp->set_training(num_ins,input,Q_action);

		netmin = env.qapp->get_min();
		netmax = env.qapp->get_max();





	}

	void precompute(){

		ind_j = new int[2*n_features];
		sign_j = new int[2*n_features];


					/*
					 *  ind_j[j] contains i%n_features
					 *  sign_j[j] contains +1 if j<n_features -1 otherwise
					 */
					for(int j=0; j< 2*n_features; j++){

						ind_j[j] = j%n_features;
						if(j<n_features){
							sign_j[j] = 1;
						}
						else {

							sign_j[j] = -1;
						}
					}

	}

	double loss(VectorXd& w_found, AppEnvironment& env, Demonstration& D) {

		double rep = 0;
		ind = 0;

		for (nep = 0; nep < D.size(); nep++) {

			for (ns = 0; ns < D[nep].size(); ns++) {

				if (D[nep][ns].s_next <= -1) {
					continue;
				}

				rep += std::pow(
						(double)Q(ind)
								- env.get_q(w_found, nep, ns, DEMO_NEXT,
										D[nep][ns].s, D[nep][ns].a), 2);

				ind++;

			}

		}

		return rep;
	}

	virtual void update(VectorXd& theta, AppEnvironment& env, Demonstration& D,
			ApproximateMDP& mdp, VectorXd* pi = NULL, VectorXd* b = NULL) {

		Q(iter) = env.get_r(mdp.R, nep, ns, DEMO_NEXT, s, a, s_next)
				+ gamma2 * env.get_qmax(*pi, nep, ns, DEMO_NEXT, s_next);

		int a3 = 0;

		if(D[nep].size()<ns+1)

		a3 = D[nep][ns+1].a;

		//Q(iter) = env.get_r(mdp.R, nep, ns, DEMO_NEXT, s, a, s_next)
		//			+ gamma2 * env.get_q(*pi, nep, ns, DEMO_NEXT, s_next,a3);











		/*outmax = Q.maxCoeff();
		outmin = Q.minCoeff();
		if(std::abs(outmax-outmin) >= 1)
		Q=(Q.array()-Q.mean())/(outmax-outmin);
		else {

			if(std::max(std::abs(outmax),std::abs(outmin)) >= 1 ){

				Q = (Q.array()-Q.mean())/(std::max(std::abs(outmax),std::abs(outmin)));
			}

			else {
				Q=(Q.array()+0.5);
			}
		}*/

		/*outmin = Q.minCoeff();
		outmax = Q.maxCoeff();
		assert(outmax!=outmin);
		A= (netmax-netmin)/(outmax-outmin);
		B = - (outmin*(netmax-netmin)/(outmax-outmin))+netmin;
		for(int i=0; i< Q.rows(); i++){
			Q(iter)=A*Q(iter)+B;
		}*/
		//Q_action(iter) = a;
		//ind++;
	}

	virtual void optGSS(VectorXd& theta, AppEnvironment& env,
			Demonstration& D) {

		double best = 100000, cur = 0;
		int iter = 0;
		clock_t endwait = clock() + max_time * CLOCKS_PER_SEC;

		VectorXd w = VectorXd::Zero(n_features);
		VectorXd w_found = VectorXd::Zero(n_features);

		Random2 rnd(n_features, 2);
		VectorXd w_best;

		double deltak = 1;
		bool improv = false;
		double actu = 500000, phik = 1, thetak = 0.5;
		int direction = 0;
		double prev;

		for (int i = 0; i < n_iterations; i++) {

			if (clock() >= endwait)
				break;
			iter = i;

			std::cout<<"Iter "<<i<<"best = "<<best<<std::endl;

			improv = false;

			/// Looking for the direction with improvement

			for (int j = 0; j < 2 * n_features; j++) {

				//w = w_found;

				//if (j < n_features) {

				//	w(j % n_features) += deltak;
				//} else {

				//	w(j % n_features) -= deltak;
				//}

				prev = w_found(ind_j[j]);
				w_found(ind_j[j])+=(sign_j[j]*deltak);


				/// Computing the loss function for this direction

				cur = loss(w_found, env, D);


				w_found(ind_j[j])=prev;

				if (cur < actu) {

					improv = true;
					actu = cur;
					direction = j;

				}

			}

			if (improv) {

				//if (direction < n_features) {

					//w_found(direction % n_features) += deltak;
				//}

				//else {

					//w_found(direction % n_features) -= deltak;
				//}

				w_found(ind_j[direction])+=(sign_j[direction]*deltak);

				phik = (rnd.generateUnity(1, false, 1, 2))(0);

				deltak = phik * deltak;

			}

			else {

				thetak=(rnd.generateUnity(1, false, 0, 1))(0);

				deltak = thetak * deltak;

			}

			if (actu < best) {

				best = actu;
				w_best = w_found;
			}

			/// Resetting the computation after 5000 iterations

			if (i % freq == 0) {
				actu = 500000;
				w_found = rnd.generateUnity(n_features, true, 0, 1);

				deltak = (rnd.generateUnity(1, false, 0, 1))(0);
				phik = (rnd.generateUnity(1, false, 1, 2))(0);
				thetak = (rnd.generateUnity(1, false, 0, 1))(0);
				/*w_found = rnd.generateUnity(n_features, true, 0, 50);
				 deltak = (rnd.generateUnity(1, false, 0, 50))(0);
				 phik = (rnd.generateUnity(1, false, 1, 50))(0);
				 thetak=(rnd.generateUnity(1, false, 0, 1))(0);*/
			}

		}
		theta = w_best;
	}

	void optLM(VectorXd& theta, AppEnvironment& env,
			Demonstration& D){




		if(env.MultipleNN){

			//VectorXd Qbis()


			theta = env.qapp->fit(Q);
		}
		else {

			//std::cout<<"Q "<<Q<<std::endl;

			theta = env.qapp->fit(Q);

		}


	}



	virtual void finalize(VectorXd& theta, AppEnvironment& env,
				Demonstration& D, ApproximateMDP& mdp) {

		optGSS(theta,env,D);
		//std::cout<<Q.transpose()<<std::endl;
		//std::cout<<"theta "<<theta.transpose()<<std::endl;
		//optLM(theta,env,D);

	}
	virtual ~FittedQEval() {

	}
};

#endif /* FITTEDQEVAL_H_ */
