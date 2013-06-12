/*
 * MyTypes.h
 *
 *  Created on: 7 Sep 2012
 *      Author: yedtoss
 */

#ifndef MYTYPES_H_
#define MYTYPES_H_

#include <iostream>
#include <map>
#include <vector>
#include <Eigen/Dense>
#include <limits>
#include "mpreal.h"
#include "MPRealSupport"

typedef Eigen::VectorXd VectorXd;
typedef Eigen::MatrixXd MatrixXd;
typedef std::vector<std::vector<double> > vdd;
typedef std::vector<double> vd;
typedef std::vector<int> vi;
typedef std::vector<std::vector<int> > vii;

typedef mpfr::mpreal mpreal;
typedef Eigen::Matrix<mpreal,Eigen::Dynamic,Eigen::Dynamic> MatrixXmp;
typedef Eigen::Matrix<mpreal,Eigen::Dynamic,1> VectorXmp;

mpreal EPS2 = 1e-20;




struct ApproximateMDP{

	VectorXd R;
	VectorXd V;
	VectorXd Q;
};


/// Creating observation on the form s,a,s_next

struct Observation{

	Observation(int s_, int a_, int s_next_){
		s = s_;
		a = a_;
		s_next = s_next_;
	}
	int s;
	int a;
	int s_next;
};

struct Episode{

	std::vector<Observation> ep;
	void NewObservation(int s, int a, int s_next){
		Observation tmp(s, a, s_next);
		ep.push_back(tmp);


	}

	int size(){

		return ep.size();
	}

	Observation& operator [](int i){
		return ep[i];
	}
};



struct Demonstration{

	std::vector<Episode> tr;
	int num_demo;
	Demonstration(){
		num_demo = 0;
	}
	void Observe(int s, int a, int s_next = 0){


		if(tr.size()<=0){
			NewEpisode();
		}

		tr[tr.size()-1].NewObservation(s, a, s_next);

		num_demo++;
	}

	void NewEpisode(){
		Episode tmp;
		tr.push_back(tmp);
	}

	int size(){
		return tr.size();
	}

	Episode& operator [] (int i){
		return tr[i];
	}
};

struct Statistics {

	int n_comp;
	double reward;
	int num_loss;
	int num_draw;
	int n_games;
	std::string name;
};




/// The following class represented a compress demonstration type

struct States{

    int a,num;

    States(int a_,int num_){

        a = a_;
        num = num_;
        //std::cerr<<a<<"\t";
    }

};

struct OneState{
    int s;
    int tot_act;
    std::vector<States> st;
    //std::vector<VectorXmp> coeff_dot;
    MatrixXmp coeff_dot;
    VectorXmp num_vec;
    OneState(int s_){

        s = s_;
        tot_act = 0;
    }

    void add(int a,int num){

        st.push_back(States(a,num));
        tot_act+=num;


    }


    int size(){
		return st.size();
	}

	States& operator [] (int i){
		return st[i];
	}





};

struct Demo_C{

    std::vector<OneState> tr;
    int num_demo;
    VectorXmp grad;
	Demo_C(){
		num_demo = 0;
	}

	void add(int s){

	    tr.push_back(OneState(s));
	    num_demo++;

	}

	int size(){
		return tr.size();
	}

	OneState& operator [] (int i){
		return tr[i];
	}

};

Demo_C compress(Demonstration &D){


    //double num = 0;
    Demo_C comp;
    //std::map<int, double> muE; // Contains the frequency of state s in the Demonstration
	std::map<std::pair<int, int>, int> piE; // Contains the frequency at which the Demonstration
                                                // takes action a at state s

		// muE[s] should contains frequency of visit for state s
		for (int i = 0; i != D.size(); ++i) {
			//Trajectory<int, int>& trajectory = D[i];

			for (int t = 0; t != (D)[i].size(); ++t) {
				int s = (D)[i][t].s;
				int a = (D)[i][t].a;


				piE[std::make_pair(s, a)] += 1;
				//num++;

			}

		}

		int last = -1;
		int i = -1;

		for (std::map<std::pair<int, int>, int>::iterator it = piE.begin();
				it != piE.end(); it++) {


               if(it->first.first != last){


                comp.add(it->first.first);
                i++;
                last = it->first.first;
                comp[i].add(it->first.second, it->second);
               }

               else{

                comp[i].add(it->first.second,it->second);
               }

        }

        return comp;




}


enum IRL_MODE {IRL_CJ, IRL_CJW, IRL_CJA, IRL_CPRB, IRL_CRPB};
enum DEMO_MODE{DEMO_CURRENT, DEMO_NEXT, DEMO_ACT_CURRENT, DEMO_ACT_NEXT, DEMO_NONE};
enum IRL_ACTIVATION{ACT_LOGSIG, ACT_TANH};
enum ALG{ALG_LRPB,ALG_LPRB,ALG_NNPRB,ALG_NNRPB, ALG_LRP, ALG_LPR, ALG_OPT, ALG_MANUAL};
enum Env{Env_BL,Env_TTT};
enum App{App_Linear, App_NN};
enum OPT{OPT_GSS,OPT_LRS,OPT_MC,OPT_NLOPT, OPT_SOLIS, OPT_GD, OPT_LRP};

enum TTT_PLAYER{TTT_PE= 0, TTT_PX = 1, TTT_PO = 2};
enum TTT_FEAT_TYPE{TTT_SINGLET=0, TTT_DOUBLET=1, TTT_TRIPLET=2, TTT_OTHER=3};

struct TTT_FEAT{
    TTT_FEAT_TYPE type;
    TTT_PLAYER player;
};
struct TTT_CASE{

    TTT_PLAYER player;
    int case_id;

};

TTT_PLAYER ttt_int2player(int val){

    if(val == TTT_PX){
        return TTT_PX;
    }
    else if(val == TTT_PO){
        return TTT_PO;
    }
    else if(val == TTT_PE){
        return TTT_PE;
    }
    else{

        std::cerr<<"Player id number "<<val<<" not known"<<std::endl;
        exit(-1);
        return TTT_PE;
    }
}



#endif /* MYTYPES_H_ */
