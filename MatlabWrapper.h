/*
 * MatlabWrapper.h
 *
 *  Created on: 10 Sep 2012
 *      Author: yedtoss
 */

#ifndef MATLABWRAPPER_H_
#define MATLABWRAPPER_H_


#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include "IRL.h"
#include "STEnvironment.h"
#include "LRP.h"
#include "Utils.h"



void exec(App approximateur = App_NN, IRL_MODE mode = IRL_CPRB,
		int max_time = 720, int n_iterations = 2000, int freq = 2000,
		OPT optimisateur =OPT_GSS,bool useMean = false,
        int num_hidden = 1, vi layerlist = vi(1,5), int reg = 2, std::string root="",bool use_kernel=false, double kernel_sig= 1,int max_points = INT_MAX, int features_type= 2, ALG alg= ALG_LPR) {

	/// Loading the environment parameters from Matlab

	std::string p = "/home/yedtoss/Documents/Memoire/files/";
	p = root+"files/";

	std::ifstream dim((p + "dim.txt").c_str()), demo((p + "demo.txt").c_str()), features((
			p + "fe.txt").c_str()), transition((p + "tr.txt").c_str()),
			successor((p + "succ.txt").c_str()),
			randdemo((p + "randdemo.txt").c_str());


	int n_states, n_actions, n_f, N, T, s, a, s_next, num_t, Nlspi;

	dim >> n_states >> n_actions >> n_f >> N>> T >> num_t >> Nlspi;
    dim.close();

    /// Demonstration

    Demonstration D;
    Demonstration Dlspi;

    std::cout<<"Observing demonstrations"<<std::endl;

    for(int i = 0; i<N; i++){

    	demo >> s >> a >> s_next;
    	D.Observe(s, a, s_next);
    }
    demo.close();


     for(int i = 0; i<Nlspi; i++){

    	randdemo >> s >> a >> s_next;
    	Dlspi.Observe(s, a, s_next);
    }
    randdemo.close();

    std::cout<<"Observing Features"<<std::endl;

    /// Constructing the state features

    std::vector<VectorXd> v_basis(n_states);
    std::vector<std::vector<VectorXd> >q_basis(n_states,
    		std::vector<VectorXd>(n_actions));
    double v = 0;
    VectorXd tmp = VectorXd::Zero(n_f);

    for(int i=0; i< n_states; i++){

    	for(int j =0; j< n_f; j++){

    		features >> v;
    		tmp(j) = v;

    	}
    	v_basis[i] = tmp;
    }

    features.close();

    std::cout<<"Observing transition"<<std::endl;

    /// Constructing the state transition probability and q_basis

    std::vector<std::vector<std::vector<double> > > tr(n_states,
    		std::vector<std::vector<double> >(n_actions,
    				std::vector<double>(num_t,0)));

    std::vector<std::vector<std::vector<int> > > succ(n_states,
        		std::vector<std::vector<int> >(n_actions,
        				std::vector<int>(num_t,0)));

    for(int i = 0; i< n_states; i++){



    	for(int j =0; j< n_actions;j++){

    		tmp = VectorXd::Zero(n_f);

    		for(int k =0; k<num_t;k++){

    			transition >> v;
    			tr[i][j][k] = v;
    			successor >> succ[i][j][k];

    			tmp += v*v_basis[succ[i][j][k]];


    		}

    		q_basis[i][j] = tmp;
    	}
    }
    transition.close();
    successor.close();

    std::cout<<"Starting environment"<<std::endl;


    /// Starting the environement

    STEnvironment env_ste(n_states, n_actions, n_f, num_t, v_basis,
            q_basis, succ, tr, use_kernel,&D, kernel_sig, max_points, features_type, approximateur, mode, useMean, num_hidden, layerlist);

    std::cout<<"Computing DFIRL"<<std::endl;


    IRL irl_ste(env_ste, D,mode,&Dlspi,reg);
    VectorXd w_found;
    std::cout<<"Computation started"<<std::endl;

    //IRL_MODE mode = IRL_CPRB;

    //irl_ste.computeMC(w_found,720,500000,mode);

    if(optimisateur == OPT_LRP){

                computeLRP(D, env_ste,w_found,alg);
    }
    else if(optimisateur == OPT_NLOPT){
    	std::cerr<<"This option has been disabled"<<std::endl;
    					exit(0);
    	//computeOpt(w_found, env_ste,D, max_time, n_iterations, 20000, 0.002,mode, &Dlspi);
    	}
    else
    irl_ste.optimize(w_found, max_time, n_iterations, mode, freq, optimisateur,false);

    //std::cout<<"Weight found"<<w_found<<std::endl;

    /// Sending the data to Matlab

    std::ofstream p_out((p+"p_out.txt").c_str()), q_out((p+"q_out.txt").c_str()),
    		 v_out((p+"v_out.txt").c_str()),time_out((p
    				+"time_out.txt").c_str());

    std::ofstream r_out;

    if(mode == IRL_CRPB){

    	r_out.open((p+"r_out.txt").c_str());
    }
    else{
    	r_out.open((p+"r_out.txt").c_str());
    }

    //vdd &r2=irl_ste.get_r();
    vdd r2=vdd(n_states,std::vector<double>(n_actions,0));
    vdd q2;
    vd v2;
    vi p2;
    int time2;

    if(optimisateur == OPT_LRP)
    {
        r2 = get_q(w_found,env_ste);
        // Probabilistic policy
        q2 = get_policy(w_found,env_ste);

        v2 = get_v(w_found, env_ste);

        p2 = get_policy_greedy(w_found,env_ste);
        time2 = -1;


    }
    else
    {
    q2=irl_ste.get_q();

    if(mode == IRL_CPRB){

        r2=q2;
    }
    else {

        r2 = irl_ste.get_r();
    }


    /// Returning probabilistic policy in q2

    q2 = irl_ste.get_policy();
    v2=irl_ste.get_v();
    p2 = irl_ste.get_policy_greedy();
    //vdd &p2 = irl_ste.get_policy();
    time2 = irl_ste.time_spent;
    }



    time_out<<time2<<std::endl;


    for(int i =0; i< n_states; i++){

    	v_out<<std::setprecision(99)<<v2[i]<<std::endl;
    	p_out<<p2[i]<<std::endl;
    	if(mode == IRL_CRPB || mode ==IRL_CPRB){
    		r_out<<r2[i][0];
    	}

    	q_out<<q2[i][0];
    	//p_out<<p2[i][0];

    	for(int j=1; j< n_actions;j++){

    		if(mode == IRL_CRPB || mode == IRL_CPRB){
    			r_out<<"\t"<<std::setprecision(99)<<r2[i][j];
    		}


    		q_out<<"\t"<<std::setprecision(99)<<q2[i][j];
    		//p_out<<"\t"<<std::setprecision(99)<<p2[i][j];

    	}
    	if(mode == IRL_CRPB || IRL_CPRB){
    		r_out<<std::endl;
    	}

    	q_out<<std::endl;

    	//p_out<<std::endl;


    }

    v_out.close();
    p_out.close();
    q_out.close();
    r_out.close();
    time_out.close();

    // End












}



#endif /* MATLABWRAPPER_H_ */
