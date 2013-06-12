/*
 * IRL.h
 *
 *  Created on: 9 Sep 2012
 *      Author: yedtoss
 */

#ifndef IRL_H_
#define IRL_H_

#include "AppEnvironment.h"
#include "MyTypes.h"
#include "ApproximatePolicyIteration.h"
//#include "ApproximateValue.h"
#include "Approximator.h"
//#include "GQ.h"
//#include "GTD.h"
#include "IRL.h"
#include "LSPI.h"
#include "GreedyGQ.h"
#include "OffPac.h"
#include "LSTDQ.h"
#include "LinearApproximator.h"
#include "Random2.h"
#include "TDLearning.h"
#include "FittedQIteration.h"
//#include "Utility.h"
#include <time.h>
#include <fstream>

#include "Matrix.h"
#include "Environment.h"
#include "MersenneTwister.h"
#include "Distribution.h"
//#include "Demonstration.h"

//#include "dlib/optimization.h"
#include <iostream>
#include <map>
//typedef dlib::matrix<double, 0, 1> column_vector;

class IRL {
public:
	//Distribution *prior;
	//ApproximatePolicyIteration lspi;
	LSPI *lspi;
	FittedQIteration * fqiter;
	AppEnvironment* env;
	Demonstration* D, *Dlspi;
	GreedyGQ *greedygq;
	OffPac *offpac;
	IRL_MODE mode;

	//ApproximatePolicy opt_policy, demo_policy;
	VectorXd opt_policy; //demo_policy;
	double p_prior, p_demo; // Keep the prior and  the Demonstration probability resp
	std::map<int, double> muE; // Contains the frequency of state s in the Demonstration
	std::map<std::pair<int, int>, double> piE; // Contains the frequency at which the Demonstration
											   // takes action a at state s
	//GQ gq; // In order to compute the policy of the Demonstration under the current reward
	//VectorXd demo_t, opt_t;  // weight of the demo and optimal policy resp
	int n_features;
	double beta;
	double prev;
	int count3;
	int time_spent;
	VectorXd opt_rew, opt_best_policy;
	vdd r_true, q_true;
	vdd p_prob;
	vi p_true;
	vd v_true;
	App type;
	bool der;

	IRL(AppEnvironment& env_, Demonstration& D_,IRL_MODE mode_=IRL_CPRB,Demonstration* Dlspi_=NULL, int reg =2) {
		beta = 1;
		lspi = new LSPI(reg);
		greedygq = new GreedyGQ();
		offpac = new OffPac();
		
		if(Dlspi_== NULL)
		Dlspi = &D_;
		else
		Dlspi = Dlspi_;
	

		if(mode_ == IRL_CRPB && env_.type == App_Linear)
		lspi->policy_eval.static_settings(env_,*Dlspi);
		fqiter = new FittedQIteration();
		fqiter->policy_eval.static_settings(env_,*Dlspi);
		count3 = 0;
		env = &env_;
		D = &D_;
		//D_lspi = D_lspi_;
		mode = mode_;
		time_spent = 0;

		n_features = env->get_num_training();

		type = env->type;

		computeFrequency();

		//computeMDP();

		//computeFrequency();

		//opt_t.Resize(n_features, 1);
		//opt_policy = ApproximatePolicy(env,Qapp,&opt_t);

		//lspi = LSPI(D, &mdp);
		//lspi.setw(&opt_t);

		//demo_t.Resize(n_features, 1);
		//demo_policy = ApproximatePolicy (env,Qapp,&demo_t);

		/// Demonstration policy evaluation with on-policy == true
		//gq = GQ(D,&mdp,true);
		//gq.setw(&demo_t);

	}

	void computeFrequency() {
		double num = 0;

		// muE[s] should contains frequency of visit for state s
		for (int i = 0; i != D->size(); ++i) {
			//Trajectory<int, int>& trajectory = D[i];

			for (int t = 0; t != (*D)[i].size(); ++t) {
				int s = (*D)[i][t].s;
				int a = (*D)[i][t].a;

				// Here muE[s] contains the number of times s occurs
				muE[s] += 1;
				piE[std::make_pair(s, a)] += 1;
				num++;

			}

		}



		for (std::map<std::pair<int, int>, double>::iterator it = piE.begin();
				it != piE.end(); it++) {


			/// Computing the frequency at which action it->first.second
			/// is taken at state it->first.first

			//it->second /= muE[it->first.first];

			// Do nothing it->second is the number of times s,a is seen
			it->second = it->second + 0;
			//std::cout<<"s = "<<it->first.first<<"a =  "<<it->first.second<<" fr = "<<it->second;

		}
		//std::cout<<"size = "<<piE.size()<<std::endl;

		for (std::map<int, double>::iterator it = muE.begin(); it != muE.end();
				it++) {

			// Computing muE[s]
			it->second /= num;

			//muE[s] /= num;

		}
	}


	/*
	 * Gradient of the computeJ with Log
	 *
	 */

	VectorXd gradient(){

		int prev_state = -2;
		VectorXd all_q;
		VectorXd gr=VectorXd::Zero(opt_policy.rows());
		VectorXd sumgr=gr;
		std::vector<VectorXd> all_gr;
		double sum = 0, sumsquare=0;

		// int n2 = 0;
		for (std::map<std::pair<int, int>, double>::iterator it = piE.begin();
				it != piE.end(); it++) {

			/// Checking if the prob of the action is the higher

			assert(it->first.second<env->get_n_actions(it->first.first));

			if (prev_state != it->first.first) {
				prev_state = it->first.first;
				all_q = VectorXd::Zero(env->get_n_actions(it->first.first));
				all_gr = std::vector<VectorXd>(env->get_n_actions(it->first.first),
						VectorXd::Zero(opt_policy.rows()));
				sumgr = VectorXd::Zero(opt_policy.rows());

				for (int a = 0; a < env->get_n_actions(it->first.first); a++) {

					all_q(a) = std::exp(
							env->get_q(opt_policy, -1, -1, DEMO_NONE,
									it->first.first, a));

					all_gr[a] = env->qapp->gradient(env->get_q_basis(-1, -1, DEMO_ACT_NEXT,
							it->first.first,a),opt_policy);

					sumgr+=all_gr[a];

				}

				sum = all_q.sum();
				sumsquare = sum*sum;

			}

			double gx = all_q(it->first.second)/sum;

			gr+= it->second*( (all_gr[it->first.second]*
							all_q(it->first.second)*sum-
							all_q(it->first.second)*sumgr)/sumsquare)/gx;

		}


		return gr;




	}

	/**
	 * Likelihood \sum\mu_E(\pi_E(s,a)-\pi(s,a))^2 with \pi_E(s,a) the frequency
	 */

	double computeJ() {
		double rep = 0;

		int prev_state = -2;
		VectorXd all_q;
		double sum =0;
		double v_max = 0;
		v_max = 0;
		double beta;
		beta = 20;

		// int n2 = 0;
		for (std::map<std::pair<int, int>, double>::iterator it = piE.begin();
				it != piE.end(); it++) {

			/// Checking if the prob of the action is the higher

			assert(it->first.second<env->get_n_actions(it->first.first)
					|| env->get_n_actions(it->first.first) == 0 );







			/// What if the state is the terminal state and
			/// There is no action

			if(env->get_n_actions(it->first.first) <= 0){
				continue;
			}

			// Likelihood similar to gpirl

			/*if (prev_state != it->first.first) {
				prev_state = it->first.first;
				all_q = VectorXd::Zero(env->get_n_actions(it->first.first));


				if(env->MultipleNN){


					all_q = env->get_q2(opt_policy, -1, -1, DEMO_NONE,
							it->first.first, 0);

					v_max = all_q.maxCoeff();


				}

				else {

				for (int a = 0; a < env->get_n_actions(it->first.first); a++) {

					all_q(a) =
							env->get_q(opt_policy, -1, -1, DEMO_NONE,
									it->first.first, a);

					if(all_q(a) > v_max || a == 0){

						v_max = all_q(a);
					}

				}
				}

				//sum = all_q.sum();
			}
			//assert(all_q(it->first.second)>1e-10);


			rep += (all_q(it->first.second)-v_max)*(it->second);

			continue;*/






			/// Log likelihood similar to christos

			/*if(prev_state != it->first.first){
				prev_state = it->first.first;
				all_q = VectorXd::Zero(env->get_n_actions(it->first.first));


				if(env->MultipleNN){


					all_q = env->get_q2(opt_policy, -1, -1, DEMO_NONE,
												it->first.first, 0);
					for(int a = 0; a < env->get_n_actions(it->first.first); a++){

						all_q(a) = (std::exp(beta*(double)all_q(a)));
					}
				}

				else{

			for (int a = 0; a < env->get_n_actions(it->first.first); a++) {

								all_q(a) = (std::exp(beta*env->get_q(opt_policy, -1, -1, DEMO_NONE,
										it->first.first, a)));

			}
				}

			sum = all_q.sum();
			}
			//assert(all_q(it->first.second)>1e-10);

			rep+=it->second*std::tanh(std::log((double)all_q(it->first.second)/sum));
			continue;*/

			/*
			 * If the beta in the likelihood is big enough we only need
			 * to check if the action with the maximum q corresponds to the
			 * demonstration action
			 *
			 */



			/// This is a trick to precompute all needed information for a state
			/// so that we don't recompute them for similar successive state



			/// If the previous state is different from the current do the precomputation

			if(prev_state != it->first.first){
				prev_state = it->first.first;
				all_q = VectorXd::Zero(env->get_n_actions(it->first.first));

			if(env->MultipleNN){
				//all_q = VectorXd::Zero(env->get_n_actions(it->first.first));

				all_q = env->get_q2(opt_policy, -1, -1, DEMO_NONE,
						it->first.first, 0);



			}

			else {

				for (int a = 0; a < env->get_n_actions(it->first.first); a++) {

					all_q(a) = env->get_q(opt_policy, -1, -1, DEMO_NONE,
							it->first.first, a);

				}

				sum = all_q.sum();
			}

			}
			//std::cout<<"all "<<all_q(0)<<" "<<all_q(1)<<std::endl;

			//double cur = env->get_q(opt_policy, -1, -1, DEMO_NONE,
			//		it->first.first, it->first.second);
			//std::cout<<"cur ="<<cur;

			double cur;

			cur = all_q(it->first.second);

			/*if(env->MultipleNN){

				cur = all_q(it->first.second);
			}
			else {

				cur = env->get_q(opt_policy, -1, -1, DEMO_NONE,
							it->first.first, it->first.second);

			}*/
			double tmp =0;



			for (int a = 0; a < env->get_n_actions(it->first.first); a++) {

				if (it->first.second != a) {
					//double tmp = env->get_q(opt_policy, -1, -1, DEMO_NONE,
						//	it->first.first, a);

					if(!env->isValid(it->first.first, a)){
						continue;
					}
					/*if(env->MultipleNN){
						tmp = all_q(a);
					}
					else {

						tmp = env->get_q(opt_policy, -1, -1, DEMO_NONE,
									it->first.first, a);
					}*/
					//std::cout<<"tmp "<<tmp<<" ot "<<cur<<" "<<it->first.second<<std::endl;


					tmp = all_q(a);

					//std::cout<<env->get_q_basis(-1, -1, DEMO_ACT_NEXT, it->first.first,a)!= env->get_q_basis(-1, -1, DEMO_ACT_NEXT, it->first.first,it->first.second)<<std::endl;

					// Don't penalize when two states have the same features

					if (tmp >= cur && env->isValid(it->first.first, a) &&(env->MultipleNN ||
							(env->get_q_basis(-1, -1, DEMO_ACT_NEXT, it->first.first,a)-env->get_q_basis(-1, -1, DEMO_ACT_NEXT, it->first.first,it->first.second)).squaredNorm()>1e-10)) {
					//if(tmp>=cur){
					//std::cout<<"bes = "<<tmp<<"\t";
						//rep += muE[it->first.first] * 1;  //std::pow(0.5-cur,2);
						rep -= 1;
						break;
					}
					continue;
				} else {
					continue;
				}

			}

			/// Checking only the prob of the action
			//rep += muE[it->first.first]*std::pow(env.get_policy(opt_policy, it->first.first, it->first.second)-it->second, 2);

			/// Checking only the action
			//int e = env.policy_act(opt_policy, it->first.first,1,true,true);
			//double tmp = 1;
			//if( e ==it->first.second )
			//tmp = 0;
			//rep+=tmp;
			//n2++;

		}
		//std::cout<<" n= "<<n2<<"\t";

		//rep = -beta * rep;
		//std::cerr<<"Rety\t"<<rety<<"\t";

		//return(rep/(double)piE.size());

		return (rep);
	}

	/*double computeJa(AppEnvironment& env, Demonstration<int, int>& D,
	 ApproximateMDP& mdp) {
	 gq.compute_on_policy(demo_policy, env, D, mdp);
	 double rep = 0;
	 for (std::map<std::pair<int, int>, double>::iterator it = piE.begin();
	 it != piE.end(); it++) {

	 rep += muE[it->first.first]
	 * std::pow(
	 env.get_policy(opt_policy, it->first.first,
	 it->first.second)
	 - env.get_policy(demo_policy,
	 it->first.first, it->first.second),
	 2);

	 }

	 rep = -beta * rep;

	 return rep;

	 }*/

	/**
	 * Likelihood:  (theta_E-theta)^2
	 */

	/*double computeJw(AppEnvironment& env, Demonstration<int, int>& D,
	 ApproximateMDP& mdp) {
	 GQ gq;
	 // Matrix demo_policy;
	 gq.compute_on_policy(demo_policy, env, D, mdp);
	 double rep = 0;
	 /// TODO Add this routine to the Matrix class

	 // Calculating the norm between demo_t and opt_t

	 for (int i = 0; i < env.get_n_features(); i++) {
	 rep += std::pow(demo_policy(i, 0) - opt_policy(i, 0), 2);

	 }
	 rep = -beta * rep;
	 return rep;

	 }*/

	/**
	 * Computing the loglikelihood of the reward w given the Demonstration
	 */
	double likelihood(const VectorXd& w, IRL_MODE mode = IRL_CPRB) {
		//Matrix w_copy = w;
		// p_prior = prior->log_pdf(w);

		/// Setting the MDP to the new reward
		//mdp.setRw(&w_copy);

		/// Computing the Optimal policy of the new reward
		/// the weight of opt_policy is automatically update

		if (mode == IRL_CRPB) {
			ApproximateMDP mdp;
			mdp.R = w;
			opt_policy = opt_best_policy;
			if(type == App_Linear){
				lspi->compute(opt_policy, *env, *Dlspi, mdp, false);
				//greedygq->compute(opt_policy, *env, *Dlspi, mdp);
				//offpac->compute(opt_policy, *env, *Dlspi, mdp);
				
			}

			else if(type == App_NN){
				fqiter->compute(opt_policy, *env, *Dlspi, mdp, false);
			}
			else {
				std::cerr<<"You should define a valid type"<<std::endl;
				exit(0);
			}
			//fqiter->compute(opt_policy, *env, *D, mdp, false);
		}

		else if (mode == IRL_CPRB) {
			opt_policy = w;
		}

		/*for(int i = 0; i<n_features; i++)
		 {
		 std::cerr<<opt_policy(i,0)<<"\t";
		 }
		 std::cerr<<std::endl;*/
		//opt_policy = w;
		/// Computing the probability that the Demonstration are sampled from
		/// the optimal policy
		double p_prior = 0;

		double p_demo = 0;
		p_demo = computeJ();

		//if (mode == IRL_CJ) {
		//p_demo = computeJ();
		//}

		//if (mode == IRL_CJA) {
		//p_demo = computeJa(env, D, mdp);

		//}

		//if (mode == IRL_CJW) {
		//p_demo = computeJw(env, D, mdp);
		//}

		double rep2 = -(p_prior + p_demo);
		//if (count3 % 10 == 0)
		//	std::cout << "L = " << rep2 << "\t";
		count3++;

		//return -(p_prior + p_demo);
		return rep2;

	}

	double likelihood(std::vector<double>&w, IRL_MODE mode = IRL_CPRB) {
		Eigen::Map<Eigen::VectorXd> tmp(w.data(), w.size());
		return likelihood(tmp, mode);

	}



	/**
	 *  Computing using a given optimisation algorithm
	 */


	void optimize(VectorXd& w_found, int max_time = 360, int n_iterations =
			2000, IRL_MODE mode_ = IRL_CRPB, int freq = 8000,OPT optim =OPT_GSS, bool useMean = false ){

		if(optim==OPT_GSS){

			computeGSS(w_found, max_time, n_iterations, mode_, freq, useMean);


		}

		else if(optim==OPT_LRS){

			computeLRS(w_found, max_time, n_iterations, mode_, freq, useMean);

		}

		else if(optim==OPT_MC){

			computeMC(w_found, max_time, n_iterations, mode_, freq, useMean);



		}

		else if(optim == OPT_SOLIS){

			computeSolis(w_found, max_time, n_iterations, mode_, freq, useMean);
		}

		else if(optim == OPT_GD){

			computeGradientDescent(w_found, max_time, n_iterations, mode_, freq, useMean);
		}



	}

	/**
	 * Computing the optimization using a simple Monte carlo
	 * algorithm :
	 */

	void computeMC(VectorXd& w_found, int max_time = 360, int n_iterations =
			2000, IRL_MODE mode = IRL_CRPB, int freq = 8000, bool useMean = false) {
		//Matrix w = Random.generateUnity(n_features,1);

		std::cout << "Monte Carlo Computation started" << std::endl;

		double best = 100000, cur = 0;
		int iter = 0;
		clock_t endwait = clock() + max_time * CLOCKS_PER_SEC;
		// double best = -100000, cur = 0;

		//if (mode == IRL_CJ || mode == IRL_CJA) {

		//computeFrequency(D);

		//}

		//n_features = env->get_n_features();
		n_features = env->get_num_training();
		if (mode == IRL_CPRB) {
			n_features = env->get_q_num_training();
		}
		//VectorXd w = VectorXd::Zero(n_features);
		w_found = VectorXd::Zero(n_features);
		opt_rew = VectorXd::Zero(n_features);
		opt_best_policy = VectorXd::Zero(env->get_q_num_training());
		//std::cerr << "Num T" << n_features << std::endl;
		//double diff =0;
		Random2 rnd(n_features, 2);
		VectorXd w_mean = VectorXd::Zero(env->get_q_num_training());
		rnd.generateGaussian(n_features, 0, 3);

		std::multimap<double,VectorXd> mean_best;

		for (int i = 0; i < n_iterations; i++) {
			//std::cout<< "Iter "<<i<<std::endl;
			//std::cerr<<"nf = "<<n_features<<"\t";
			//exit(0);
			if (clock() >= endwait)
				break;
			iter = i;
			if(iter%50==0)
				std::cerr<<"Iterations "<<iter<<" v= "<<best<<std::endl;
			else if(env->type==App_NN)
				std::cerr<<"Iterations "<<iter<<" v= "<<best<<std::endl;
			//VectorXd& w = rnd.generateUnity(n_features, true, 0, 2);
			//VectorXd& w = rnd.generateGaussian(n_features, 0, 0.1);
			//VectorXd& w = rnd.generateUnity(n_features, true, 0, 1);
			//if((w.cwiseAbs()).maxCoeff() > 1){
				//continue;
			//}

			//VectorXd& w = rnd.generateGaussian(n_features, 0, 3);
			VectorXd& w = rnd.generateDirichlet();
			//w = VectorXd::Ones(n_features);
			//std::cerr<<" pp\t";
			cur = likelihood(w, mode);

			/*double prior = 1;

			for(int k =0 ; k < w.rows();k++){

				prior += (1-rnd.gauss->pdf((real)w(k)));
			}

			assert(prior >= 0);

			cur += ((prior)/(w.rows()))*piE.size();*/


			if(useMean){

			if(mode == IRL_CRPB){
				//w_mean += (1-cur)*opt_policy;
				mean_best.insert(std::make_pair(1-cur,opt_policy));
			}
			else {
				//w_mean += (1-cur)*w;
				mean_best.insert(std::make_pair(1-cur,w));
			}

			if(mean_best.size()>2000){
				mean_best.erase(mean_best.begin());
			}
			}
			if (cur < best) {
				//if(cur > best) {
				best = cur;
				w_found = w;

				if (mode == IRL_CRPB) {
									opt_best_policy = opt_policy;
								}

			}

		}



		if (mode == IRL_CRPB) {
			ApproximateMDP mdp;
			mdp.R = w_found;
			opt_rew = w_found;

			//lspi->compute(w_found, *env, *D, mdp);
			w_found = opt_best_policy;

		}

		opt_policy = w_found;


		if(mode == IRL_CPRB){

			opt_rew = w_found;
		}
		if(useMean){

					//w_found = w_mean;

			for(std::multimap<double,VectorXd>::iterator it=mean_best.begin();it!=mean_best.end();it++){

				w_mean+=(it->first)*it->second;
			}
			w_found = w_mean;
		}

		time_spent = clock() / CLOCKS_PER_SEC;

		std::cout << "Weight found" << w_found.transpose() << std::endl;

		std::cout << " size = " << piE.size() << std::endl;
		std::cout << " The optimal value of the objective is : Likelihood = "
				<< best << std::endl;
		std::cout << "It  has been found after " << iter << " Iterations"
				<< std::endl;
		std::cout << "Time spent in seconds " << clock() / CLOCKS_PER_SEC
				<< std::endl;
		//std::cout<<" LikelihoodPoten = "<<b<<std::endl;

	}

	void computeBasicSPSA(VectorXd& w_found, int max_time = 360, int n_iterations =
			2000, IRL_MODE mode = IRL_CRPB, int freq = 8000, bool useMean = false){
		std::cout << "Basic simultaneous Pertubation stochastic optimization"
				" Computation started" << std::endl;

		double best = 100000, cur = 0;
		int iter = 0;
		clock_t endwait = clock() + max_time * CLOCKS_PER_SEC;
		n_features = env->get_num_training();
		if (mode == IRL_CPRB) {
			n_features = env->get_q_num_training();
		}
		opt_best_policy = VectorXd::Zero(env->get_q_num_training());
		VectorXd w = VectorXd::Zero(n_features);
		VectorXd gk;
		w_found = VectorXd::Random(n_features);
		opt_rew = VectorXd::Zero(n_features);
		Random2 rnd(n_features, 2);
		double ak, a, c, ck, A,alpha = 0.602, gamma = 0.101, cur2;

		//alpha = 1;
		//gamma = 1.0/6.0;
		A = n_iterations*0.1;
		c = 0.2; // Noise standard deviation
		double var = 5;  // Desired variance of elements in w_found
		double mag = 0.5;  // Mean Magnitude of gk of w_found0
		a = (var/mag)*std::pow(A+1,alpha);
		VectorXd w_best;
		w_found = rnd.generateUnity(n_features, true, 0, 2);
		int k = -1;

		for (int iter = 0; iter < n_iterations; iter++) {

			if (clock() >= endwait)
				break;
			k++;
			if(iter%50==0)
				std::cerr<<"Iterations "<<iter<<" v= "<<best<<std::endl;
			VectorXd& deltak = rnd.generateBernoulli();
			ak = a/std::pow(k+1+A, alpha);
			ck = c/std::pow(k+1,gamma);

			// Loss function Evaluation
			w = w_found + ck*deltak;
			cur = likelihood(w,mode);
			if(cur < best){
				w_best = w;
				best = cur;
				if (mode == IRL_CRPB) {
					opt_best_policy = opt_policy;
				}
			}

			w = w_found - ck*deltak;
			cur2 = likelihood(w,mode);
			if(cur2 < best){

				w_best = w;
				best = cur2;

				if (mode == IRL_CRPB) {
									opt_best_policy = opt_policy;
								}
			}

			// Gradient approximation
			gk = (cur - cur2)/(2.0*ck)* deltak.cwiseInverse();

			// Updating w_found

			w_found -= (ak*gk);

			// Resetting the computation after 2000 iterations
			if(k==2000){
				k = -1;
				w_found = rnd.generateUnity(n_features, true, 0, 2);
			}

		}
		w_found = w_best;


		if (mode == IRL_CRPB) {
			ApproximateMDP mdp;
			mdp.R = w_found;
			opt_rew = w_found;

			//lspi->compute(w_found, *env, *D, mdp);
			w_found = opt_best_policy;

		}

		opt_policy = w_found;

		time_spent = clock() / CLOCKS_PER_SEC;

		std::cout << "Weight found" << w_found << std::endl;

		std::cout << " size = " << piE.size() << std::endl;
		std::cout << " The optimal value of the objective is : Likelihood = "
				<< best << std::endl;
		std::cout << "It  has been found after " << iter << " Iterations"
				<< std::endl;
		std::cout << "Time spent in seconds " << clock() / CLOCKS_PER_SEC
				<< std::endl;
		//std::cout<<" LikelihoodPoten = "<<b<<std::endl;

	}

	void computeLRS(VectorXd& w_found, int max_time = 360, int n_iterations =
			2000, IRL_MODE mode = IRL_CRPB, int freq = 8000, bool useMean = false) {

		//Matrix w = Random.generateUnity(n_features,1);

		std::cout << "Localised Random Search Computation started" << std::endl;

		double best = 100000, cur = 0;
		int iter = 0;
		clock_t endwait = clock() + max_time * CLOCKS_PER_SEC;
		// double best = -100000, cur = 0;

		//if (mode == IRL_CJ || mode == IRL_CJA) {

		//computeFrequency(D);

		//}

		//n_features = env->get_n_features();
		n_features = env->get_num_training();
		if (mode == IRL_CPRB) {
			n_features = env->get_q_num_training();
		}
		VectorXd w = VectorXd::Zero(n_features);
		opt_best_policy = VectorXd::Zero(env->get_q_num_training() );
		w_found = VectorXd::Zero(n_features);
		opt_rew = VectorXd::Zero(n_features);
		//std::cerr << "Num T" << n_features << std::endl;
		//double diff =0;
		Random2 rnd(n_features, 2);
		VectorXd w_best;

		for (int i = 0; i < n_iterations; i++) {
			//std::cout<< "Iter "<<i<<std::endl;
			//std::cerr<<"nf = "<<n_features<<"\t";
			//exit(0);
			if (clock() >= endwait)
				break;
			iter = i;
			if(iter%50==0)
			std::cerr<<"Iterations "<<iter<<" v= "<<best<<std::endl;
			else if(env->type==App_NN)
						std::cout<<"Iterations "<<iter<<" v= "<<best<<std::endl;
			//VectorXd& dk = rnd.generateUnity(n_features, true, 0, 0.02);
			//VectorXd& dk = rnd.generateUnity(n_features, true, 0, 50);
			VectorXd& dk = rnd.generateGaussian(n_features, 0, 0.5);
			//VectorXd& w = rnd.generateDirichlet();
			//w = VectorXd::Ones(n_features);
			//std::cerr<<" pp\t";
			w = w_found + dk;
			//std::cout<<"rew "<<w<<std::endl;
			cur = likelihood(w, mode);
			if (cur < best) {
				//if(cur > best) {
				best = cur;
				w_found = w;
				w_best = w;
				if (mode == IRL_CRPB) {
					opt_best_policy = opt_policy;
				}

			}
			//opt_policy = w_best;
			//w = w_found;

			/// Resetting the computation after 5000 iterations

			if(i %freq ==0){
				w_found = VectorXd::Zero(n_features);
				//w_found = rnd.generateUnity(n_features, true, 0, 2);
				//w_found = rnd.generateUnity(n_features, true, 0, 50);
				w_found = rnd.generateGaussian(n_features, 0, 0.5);
			}

		}
		w_found = w_best;

		if (mode == IRL_CRPB) {
			ApproximateMDP mdp;
			mdp.R = w_found;
			opt_rew = w_found;

			//lspi->compute(w_found, *env, *D, mdp, false);
			w_found = opt_best_policy;

		}

		opt_policy = w_found;

		time_spent = clock() / CLOCKS_PER_SEC;

		std::cout << "Weight found" << w_found << std::endl;

		std::cout << " size = " << piE.size() << std::endl;
		std::cout << " The optimal value of the objective is : Likelihood = "
				<< best << std::endl;
		std::cout << "It  has been found after " << iter << " Iterations"
				<< std::endl;
		std::cout << "Time spent in seconds " << clock() / CLOCKS_PER_SEC
				<< std::endl;
		//std::cout<<" LikelihoodPoten = "<<b<<std::endl;

	}





	void computeSolis(VectorXd& w_found, int max_time = 360, int n_iterations =
				2000, IRL_MODE mode = IRL_CRPB, int freq = 8000, bool useMean = false) {

			//Matrix w = Random.generateUnity(n_features,1);

			std::cout << "Solis Search Computation started" << std::endl;

			double best = 100000, cur = 0;
			int iter = 0;
			clock_t endwait = clock() + max_time * CLOCKS_PER_SEC;
			// double best = -100000, cur = 0;

			//if (mode == IRL_CJ || mode == IRL_CJA) {

			//computeFrequency(D);

			//}

			//n_features = env->get_n_features();
			n_features = env->get_num_training();
			if (mode == IRL_CPRB) {
				n_features = env->get_q_num_training();
			}
			VectorXd w = VectorXd::Zero(n_features);
			opt_best_policy = VectorXd::Zero(env->get_q_num_training() );
			w_found = VectorXd::Zero(n_features);
			opt_rew = VectorXd::Zero(n_features);
			//std::cerr << "Num T" << n_features << std::endl;
			//double diff =0;
			Random2 rnd(n_features, 2);
			VectorXd w_best, bk=VectorXd::Zero(n_features);



			//std::ifstream t_in("/home/yedtoss/workspace/RL3/src/true_r.txt");
			for (int i = 0; i < n_iterations; i++) {
				//std::cout<< "Iter "<<i<<std::endl;
				//std::cerr<<"nf = "<<n_features<<"\t";
				//exit(0);
				if (clock() >= endwait)
					break;



//				for(int i2=0; i2 < n_features; i2++){
//
//					t_in>>w(i2);
//				}
//				std::cout<<w.transpose()<<std::endl;
//
//				cur = likelihood(w,mode);
//				best = cur;
//				w_best = w;
//				opt_best_policy = opt_policy;

				iter = i;
				if(iter%50==0)
				std::cerr<<"Iterations "<<iter<<" v= "<<best<<std::endl;
				else if(env->type==App_NN)
							std::cerr<<"Iterations "<<iter<<" v= "<<best<<std::endl;

				//continue;
				//VectorXd& dk = rnd.generateUnity(n_features, true, 0, 0.02);
				//VectorXd& dk = rnd.generateUnity(n_features, true, 0, 50);
				VectorXd& dk = rnd.generateGaussian(n_features, 0, 3);
				//VectorXd& dk = rnd.generateDirichlet();
				//VectorXd& dk = rnd.generateGaussian(n_features, 0, 0.1);
				//VectorXd& w = rnd.generateDirichlet();
				//w = VectorXd::Ones(n_features);
				//std::cerr<<" pp\t";
				w = w_found + dk +bk;

				/*if((w.cwiseAbs()).maxCoeff() >1){

					continue;
				}*/
				//std::cout<<"rew "<<w<<std::endl;

				double s2 = (w.cwiseAbs()).sum();
				if(s2<=1e-30){
					continue;
				}
				else{
					w=w/s2;
				}
				cur = likelihood(w, mode);
				if (cur < best) {
					//if(cur > best) {
					best = cur;
					w_found = w;
					w_best = w;
					bk = 0.2*bk+0.4*dk;
					if (mode == IRL_CRPB) {
						opt_best_policy = opt_policy;
					}

				}
				else {

					w = w_found +dk -bk;
					cur = likelihood(w, mode);


					if (cur < best) {
					//if(cur > best) {
					best = cur;
					w_found = w;
					w_best = w;
					bk = bk - 0.4 * dk;
					if (mode == IRL_CRPB) {
						opt_best_policy = opt_policy;
					}

				}

					else {

						bk = 0.5*bk;
					}




				}
				//opt_policy = w_best;
				//w = w_found;

				/// Resetting the computation after 5000 iterations

				if(i %freq ==0){
					w_found = VectorXd::Zero(n_features);

					//w_found = rnd.generateUnity(n_features, true, 0, 2);
					//w_found = rnd.generateUnity(n_features, true, 0, 50);
					//w_found = rnd.generateGaussian(n_features, 0, 3);
					w_found = VectorXd::Zero(n_features);
					bk = VectorXd::Zero(n_features);
				}

			}
			w_found = w_best;

			if (mode == IRL_CRPB) {
				ApproximateMDP mdp;
				mdp.R = w_found;
				opt_rew = w_found;

				//lspi->compute(w_found, *env, *D, mdp, false);
				w_found = opt_best_policy;

			}

			opt_policy = w_found;

			time_spent = clock() / CLOCKS_PER_SEC;

			std::cout << "Weight found" << w_found.transpose() << std::endl;

			std::cout << " size = " << piE.size() << std::endl;
			std::cout << " The optimal value of the objective is : Likelihood = "
					<< best << std::endl;
			std::cout << "It  has been found after " << iter << " Iterations"
					<< std::endl;
			std::cout << "Time spent in seconds " << clock() / CLOCKS_PER_SEC
					<< std::endl;
			//std::cout<<" LikelihoodPoten = "<<b<<std::endl;

		}






	void computeGSS(VectorXd& w_found, int max_time = 360, int n_iterations =
				2000, IRL_MODE mode = IRL_CRPB, int freq = 8000, bool useMean = false) {

			//Matrix w = Random.generateUnity(n_features,1);

			std::cout << "Compass Search Computation started" << std::endl;

			double best = 100000, cur = 0;
			int iter = 0;
			clock_t endwait = clock() + max_time * CLOCKS_PER_SEC;
			// double best = -100000, cur = 0;

			//if (mode == IRL_CJ || mode == IRL_CJA) {

			//computeFrequency(D);

			//}

			//n_features = env->get_n_features();
			n_features = env->get_num_training();
			if (mode == IRL_CPRB) {
				n_features = env->get_q_num_training();
			}
			VectorXd w = VectorXd::Zero(n_features);
			opt_best_policy = VectorXd::Zero(env->get_q_num_training() );
			w_found = VectorXd::Zero(n_features);
			//w_found = VectorXd::Random(n_features);
			opt_rew = VectorXd::Zero(n_features);
			//std::cerr << "Num T" << n_features << std::endl;
			//double diff =0;
			Random2 rnd(n_features, 2);
			VectorXd w_best;
			VectorXd w_mean = VectorXd::Zero(env->get_q_num_training());

			VectorXd opt_actu = VectorXd::Zero(env->get_q_num_training());
			std::multimap<double, VectorXd> mean_best;

			double deltak=1;
			bool improv =false;
			double actu = 500000, phik = 1, thetak = 0.5;
			int direction = 0;
			double prev;
			int ind_j[2*n_features];
			int sign_j[2*n_features];
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

			for (int i = 0; i < n_iterations; i++) {

				if (clock() >= endwait)
					break;
				iter = i;
				if(iter%50==1)

				std::cerr<<"Iterations "<<iter<<" v= "<<best<<std::endl;
				//std::cout<<"Opt best"<<opt_best_policy<<std::endl;

				improv = false;


				/// Looking for the direction with improvement

				for(int j=0; j< 2*n_features; j++){

					//w = w_found;

					//if(j<n_features){

						//w(j%n_features)+= deltak;
				//	}
					//else {

						//w(j%n_features) -= deltak;
					//}

					prev = w_found(ind_j[j]);

					w_found(ind_j[j]) += (sign_j[j]*deltak);

					/// Computing the loss function for this direction



					/*double s2 = (w_found.cwiseAbs()).sum();
					if(s2<=1e-30){
						w_found(ind_j[j]) = prev;
							continue;
					}
					else{
						w=w_found/s2;
					}*/

					cur = likelihood(w_found,mode);
					//cur = likelihood(w,mode);

				if (useMean) {

					if (mode == IRL_CRPB) {
						//w_mean += (1-cur)*opt_policy;
						mean_best.insert(std::make_pair(1 - cur, opt_policy));
					} else {
						//w_mean += (1-cur)*w;
						mean_best.insert(std::make_pair(1 - cur, w_found));
					}

					if (mean_best.size() > 5) {
						mean_best.erase(mean_best.begin());
					}
				}

					w_found(ind_j[j]) = prev;

					if(cur < actu){

						improv = true;
						actu = cur;
						direction = j;
						if (mode == IRL_CRPB) {
							opt_actu = opt_policy;
						}
						break;


					}

					if(mode==IRL_CRPB && env->type==App_NN){

						//std::cout<<"Intermediaire "<<actu<<std::endl;
					}


				}


				if(improv){

					//if(direction < n_features){

						//w_found(direction%n_features)+=deltak;
					//}

					//else {

						//w_found(direction%n_features)-=deltak;
					//}

					w_found(ind_j[direction]) += (sign_j[direction]*deltak);

					//phik = (rnd.generateUnity(1, false, 1, 2))(0);

					deltak = phik*deltak;


				}

				else {

					//thetak=(rnd.generateUnity(1, false, 0, 1))(0);

					deltak = thetak*deltak;
					if(deltak < 1e-30){
						deltak = 1.5;
					}


				}




				if (actu < best) {

					best = actu;
					w_best = w_found;
					if (mode == IRL_CRPB) {
						opt_best_policy = opt_actu;
					}

				}

				/// Resetting the computation after 5000 iterations

				if(i %freq ==0){
					//w_found = VectorXd::Zero(n_features);
					actu = 500000;
					w_found = rnd.generateUnity(n_features, true, 0, 1);

					//deltak = (rnd.generateUnity(1, false, 0, 1))(0);
					deltak = 1;
					//phik = (rnd.generateUnity(1, false, 1, 2))(0);
					phik = (rnd.generateUnity(1, false, 1, 1.5))(0);
					//thetak=(rnd.generateUnity(1, false, 0, 1))(0);
					thetak=(rnd.generateUnity(1, false, 0.5, 0.7))(0);
					/*w_found = rnd.generateUnity(n_features, true, 0, 50);
					deltak = (rnd.generateUnity(1, false, 0, 50))(0);
					phik = (rnd.generateUnity(1, false, 1, 50))(0);
					thetak=(rnd.generateUnity(1, false, 0, 1))(0);*/
				}

			}
			w_found = w_best;

			if (mode == IRL_CRPB) {
				//ApproximateMDP mdp;
				//mdp.R = w_found;
				opt_rew = w_found;

				//lspi->compute(w_found, *env, *D, mdp, false);
				w_found = opt_best_policy;

			}

			if(mode == IRL_CPRB){

				opt_rew = w_found;
			}



			opt_policy = w_found;

		if (useMean) {

			//w_found = w_mean;

			for (std::multimap<double, VectorXd>::iterator it =
					mean_best.begin(); it != mean_best.end(); it++) {

				w_mean += (it->first) * it->second;
			}
			w_found = w_mean;
		}

			time_spent = clock() / CLOCKS_PER_SEC;

			std::cout << "Weight found" << w_found.transpose() << std::endl;

			std::cout << " size = " << piE.size() << std::endl;
			std::cout << " The optimal value of the objective is : Likelihood = "
					<< best << std::endl;
			std::cout << "It  has been found after " << iter << " Iterations"
					<< std::endl;
			std::cout << "Time spent in seconds " << clock() / CLOCKS_PER_SEC
					<< std::endl;
			//std::cout<<" LikelihoodPoten = "<<b<<std::endl;

		}


	void computeGradientDescent(VectorXd& w_found, int max_time = 360, int n_iterations =
				2000, IRL_MODE mode = IRL_CRPB, int freq = 8000, bool useMean = false) {



		/*clock_t endwait = clock() + max_time * CLOCKS_PER_SEC;
		int iter = 0;

		std::cout<<"Starting Gradient Descent Computation"<<std::endl;

		if(mode == IRL_CRPB){

			std::cout<<"Not yet implemented"<<std::endl;
			exit(0);
		}

		n_features = env->get_q_num_training();

		w_found = VectorXd::Zero(n_features);
		opt_policy = w_found;
		double best =-5000000000;
		double t = 0.02;
		double ma, mi;




		for(int i=0; i< n_iterations; i++){


			if (clock() >= endwait)
				break;

			iter =i;

			VectorXd gr = gradient();




			ma = gr.maxCoeff();
			mi = gr.minCoeff();

			if(std::abs(ma)!=std::abs(mi))


			t = 1.0/(std::max(std::abs(ma),std::abs(mi)));

			else

				t = 0.0002;



			opt_policy = opt_policy + t*gr;

			best = computeJ();

			if(iter%200==1){

			std::cout<<"Iterations "<<i<<" best = "<<best<<std::endl;
			std::cout<<gr<<std::endl;
			}









		}
		w_found = opt_policy;


		time_spent = clock() / CLOCKS_PER_SEC;

					std::cout << "Weight found" << w_found << std::endl;

					std::cout << " size = " << piE.size() << std::endl;
					std::cout << " The optimal value of the objective is : Likelihood = "
							<< best << std::endl;
					std::cout << "It  has been found after " << iter << " Iterations"
							<< std::endl;
					std::cout << "Time spent in seconds " << clock() / CLOCKS_PER_SEC
							<< std::endl;*/







	}

	std::vector<double> gradient2(){



		VectorXd tmp=gradient();
		std::vector<double> es(n_features,0);

		for(int i=0; i< n_features;i++){

			es[i] = tmp(i);
		}

		return es;
	}

	double moi(std::vector<double>& arg){

		for(int i=0; i< opt_policy.rows();i++){

					opt_policy(i) = arg[i];
				}

				return computeJ();

	}


	vdd& get_r() {

		//r_true.resize(env->get_n_states(),
			//	std::vector<double>(env->get_n_actions()));

		r_true = vdd(env->get_n_states(),
				std::vector<double>(env->get_n_actions(),0));

		for (int i = 0; i < env->get_n_states(); i++) {

			for (int j = 0; j < env->get_n_actions(); j++) {

				r_true[i][j] = env->get_r(opt_rew, -1, -1, DEMO_NONE, i, -1);
			}
		}

		return r_true;

	}

	vd& get_v() {

		//v_true.resize(env->get_n_states());
		v_true = vd(env->get_n_states());
		if (q_true.size() == 0) {
			get_q();
		}

		for (int i = 0; i < env->get_n_states(); i++) {

			v_true[i] = *std::max_element(q_true[i].begin(), q_true[i].end());

		}

		return v_true;
	}

	vi& get_policy_greedy() {

		//p_true.resize(env->get_n_states());

		p_true = vi(env->get_n_states());

		for (int i = 0; i < env->get_n_states(); i++) {

			p_true[i] = env->policy_act(opt_policy, i, true);
		}

		return p_true;

	}

	vdd & get_policy(){

		p_prob = vdd(env->get_n_states(),std::vector<double>(env->get_n_actions(),0));



		if(q_true.size()<=0){

			get_q();
		}
		double sum = 0;

		for(int s=0; s< env->get_n_states(); s++){

			for(int a=0; a<env->get_n_actions();a++){

				sum+=std::exp(q_true[s][a]);
			}

			for(int a=0; a<env->get_n_actions();a++){

				p_prob[s][a]=std::exp(q_true[s][a])/sum;
			}



		}

		return p_prob;
	}

	vdd& get_q() {

		//q_true.resize(env->get_n_states(),
			//	std::vector<double>(env->get_n_actions()));

		q_true = vdd(env->get_n_states(),
				std::vector<double>(env->get_n_actions(),0));

		for (int i = 0; i < env->get_n_states(); i++) {

			for (int j = 0; j < env->get_n_actions(); j++) {

				q_true[i][j] = env->get_q(opt_policy, -1, -1, DEMO_NONE, i, j);
			}
		}

		return q_true;

	}

	/**
	 * Computing the optimization : the maximum of the likelihood
	 * using the derivative-free algorithms bobyqa
	 */

	/*void computebobyqa(Matrix& w_found, AppEnvironment& env,
	 Demonstration<int, int>& D, int n_iterations = 20000,
	 IRL_MODE mode = IRL_CJ) {
	 Db = &D;
	 envb = &env;
	 modeb = mode;

	 if (mode == IRL_CJ || mode == IRL_CJA) {

	 computeFrequency(D);

	 }
	 n_features = env.get_n_features();

	 column_vector starting_point;
	 starting_point.set_size(n_features);
	 for (int i = 0; i < n_features; i++) {
	 starting_point(i) = ((double) (rand() % 20)) / 23.0;
	 starting_point(i) = 0.00;
	 }
	 //starting_point =-1,-1,0.11558,-1;

	 //
	 // n_features + 2 <= npt <= (n_features+1)*(n_features+2)/2
	 //

	 // Take npt to be the mean of the interval
	 // long npt =( (n_features+1)*(n_features+2)/2 + n_features + 2 ) /2;

	 //  0 < rho_end < rho_begin
	 // min(x_upper - x_lower) > 2*rho_begin
	 //
	 //double rho_begin =0.9200 , rho_end = 2e-20;
	 //prev = 200;
	 //
	 // From dlib library


	 //dlib::find_min_using_approximate_derivatives(dlib::cg_search_strategy(),
	 //                            dlib::gradient_norm_stop_strategy(1e-275).be_verbose(),
	 //                          *this, starting_point, 0);

	 dlib::find_min_bobyqa(*this,   // Function to maximize
	 starting_point,
	 npt,                        // number npt of interpolation points such that
	 dlib::uniform_matrix<double>(n_features,1, -1),  // lower bound constraint
	 dlib::uniform_matrix<double>(n_features,1, 1),   // upper bound constraint
	 rho_begin,    // initial trust region radius
	 rho_end,  // stopping trust region radius
	 5000    // max number of objective function evaluations
	 );

	 w_found = convert2(starting_point);
	 std::cout<<"size = "<<piE.size()<<std::endl;
	 std::cout<<" Likelihoodbo = "<<likelihood(starting_point)<<std::endl;

	 }*/
	virtual ~IRL() {

	}
	;
	//double essr(column_vector et) {
	//return 2;

	//}

	/*double operator()(column_vector& arg) {
	 // return the mean squared error between the target vector and the input vector
	 //column_vector tmp = arg;
	 //  double r = likelihood(arg);
	 return likelihood(arg);
	 double rep = 0;
	 for (int i = 0; i < arg.size(); i++) {
	 rep += std::abs(arg(i));
	 }
	 return rep;
	 //return arg(2)-arg(0);
	 //return essr(arg);
	 // int t = 2;
	 //return t;
	 }*/
protected:
private:
};

#endif /* IRL_H_ */
