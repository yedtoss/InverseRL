/*
 * LSTDQ.h
 *
 *  Created on: 8 Sep 2012
 *      Author: yedtoss
 *
 *      This is the version of LSTD with "exact" matrix inverse
 */

#ifndef LSTDQ_H_
#define LSTDQ_H_

#include <cmath>
#include "TDLearning.h"

class LSTDQ : public TDLearning {
public:

	VectorXd z;
	MatrixXd B,Z;
	int s_next_r;
	int a_t;
	bool check2;
	Eigen::FullPivLU<MatrixXd> lu2;
	int reg;
	LSTDQ(int reg_ =2, double gamma_ = 1, bool fast_ = false){

		init(gamma_,fast_);
		reg = reg_;

	}

	/// Don't forget not to mess init and initialize because
	/// initialize must be called inside based class

	void init(double& gamma_, bool fast_){
		gamma2 = gamma_;
		fast =fast_;
	}

	void initialize(VectorXd& theta){

		//B.resize(n_features,n_features);
		//B = MatrixXd::Zero(n_features,n_features);
		//B = 0.02*MatrixXd::Ones(n_features,n_features);


		if(!fast){
		B= reg*MatrixXd::Identity(n_features,n_features);

		//std::cout<<B<<std::endl;
		//exit(0);

		theta = VectorXd::Zero(n_features);
		//theta.resize(n_features);
		z = VectorXd::Zero(n_features);
		}
		//z.resize(n_features);
	}

	virtual void static_settings(AppEnvironment& env, Demonstration& D) {

		/// We are using  only policy evaluation
		/// so we can precompute B and Z = \phi_q \phi_r'

		if (fast) {



			Z = MatrixXd::Zero(env.get_q_num_training(),env.rapp->get_num_training());

			B= reg*MatrixXd::Identity(env.get_q_num_training(),
					env.get_q_num_training());

			for (nep = 0; nep < D.size(); nep++) {

				for (ns = 0; ns < D[nep].size(); ns++) {

					s = D[nep][ns].s;
					a = D[nep][ns].a;
					s_next = D[nep][ns].s_next;
					if (s_next <= -1) {
						continue;
					}

					VectorXd& phi = env.get_q_basis(nep, ns, DEMO_CURRENT, s,
							a);
					//VectorXd& phi_next = env.get_q_basis(nep, ns, DEMO_ACT_NEXT, s_next,env.policy_act(
					//	*pi, s_next, true, nep, ns, DEMO_ACT_NEXT) );

					a_t = 0;

					if (ns + 1 != D[nep].size()) {

						a_t = D[nep][ns + 1].a;
					}

					VectorXd& phi_next = env.get_q_basis(nep, ns, DEMO_ACT_NEXT,
							s_next, a_t);

					VectorXd & phi_r = env.get_v_basis(nep, ns, DEMO_ACT_NEXT,
							s, a, s_next);

					B = B + phi * ((phi - gamma2 * phi_next).transpose());

					Z = Z + phi*phi_r.transpose();


				}
			}

			lu2.compute(B);

		}

	}

	virtual void update(VectorXd& theta, AppEnvironment& env, Demonstration& D, ApproximateMDP& mdp,
					VectorXd* pi = NULL, VectorXd* b = NULL){

		VectorXd& phi = env.get_q_basis(nep, ns, DEMO_CURRENT, s, a);
		VectorXd& phi_next = env.get_q_basis(nep, ns, DEMO_ACT_NEXT, s_next,env.policy_act(
				*pi, s_next, true, nep, ns, DEMO_ACT_NEXT) );

		a_t = 0;

		if(ns+1 != D[nep].size()){

			a_t =D[nep][ns+1].a;
		}
		//std::cout<<"b"<<" ";

		//VectorXd& phi_next = env.get_q_basis(nep, ns, DEMO_ACT_NEXT, s_next,a_t);

		B = B + phi*((phi-gamma2*phi_next).transpose());
		z = z + phi*env.get_r(mdp.R, nep, ns, DEMO_NEXT, s, a, s_next);
	}

	virtual void finalize(VectorXd& theta, AppEnvironment& env,  Demonstration& D, ApproximateMDP& mdp){
		//std::cout<<B<<std::endl;


		//theta = B.colPivHouseholderQr().solve(z);
		//theta = B.fullPivHouseholderQr().solve(z);
		//Eigen::JacobiSVD<MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
		//theta =svd.solve(z);
		//std::cout<<"B "<<B<<std::endl;
		//std::cout<<"Z "<<z<<std::endl;
		//std::cout<<"sol "<<theta<<std::endl;
		//std::cout<<"error" <<(B)*(theta)-z<<std::endl;
		//exit(0);

		if(fast){

			theta = lu2.solve((Z*mdp.R));
			//std::cout<<"Error "<<((B*theta-Z*mdp.R).cwiseAbs()).sum()<<std::endl;



		}

		else {
			theta = B.fullPivLu().solve(z);
		}



		//theta = B.partialPivLu().solve(z);
		/*check2 = z.isApprox(B*theta);

		if(!check2){
			std::cout<<"B "<<B<<std::endl;
			std::cout<<"Z "<<z.transpose()<<std::endl;
			std::cout<<theta.transpose()<<std::endl;
			std::cout<<"Error "<<((B*theta-z).cwiseAbs()).sum()<<std::endl;
			std::cout<<"B*theta "<<(B*theta).transpose()<<std::endl;
			std::cout<<"z "<<z.transpose()<<std::endl;
			assert(check2);
		}*/


		//std::cout<<"z"<<z<<std::endl;
		//std::cout<<"theta"<<theta<<std::endl;
		//exit(0);
		/*for(int i = 0; i<n_features; i++){
			assert(isinf((double)theta[i]) == 0);
			assert(isnan((double)theta[i]) == 0);
		}*/
	}


	virtual ~LSTDQ(){

	}
};

#endif /* LSTDQ_H_ */
