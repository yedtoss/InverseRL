/*
 * AppBlackjack.h
 *
 *  Created on: 10 Sep 2012
 *      Author: yedtoss
 */

#ifndef APPBLACKJACK_H_
#define APPBLACKJACK_H_


#include "Matrix.h"
//#include "Environment.h"
#include "MersenneTwister.h"
#include "Distribution.h"
//#include "Demonstrations.h"
#include "DiscreteMDP.h"
//#include "Inc.h"
#include "Utils.h"

#include <boost/random/random_device.hpp>
#include <boost/random/uniform_int_distribution.hpp>

#include "AppEnvironment.h"
boost::random::random_device gen;

class AppBlackjack: public AppEnvironment {
public:

	DiscreteMDP* mdp;
	std::vector<VectorXd> v_basis;
	std::vector<std::vector<VectorXd> > q_basis;
	std::vector<VectorXmp> v_basis_bis;
	std::vector<std::vector<VectorXmp> > q_basis_bis;


	//AppEnvironment ess;

    AppBlackjack(App type_, bool use_kernel_=false, Demonstration *D_=NULL, IRL_MODE mode = IRL_CPRB , int num_hidden = 1, vi layerlist = vi(1,5), bool MultipleNN_ = false, double win_value_ = 1.0, double draw_value_ = 0.0,
			double loss_value_ = -1.0) {
		//ctor
		win_value = win_value_;
		draw_value = draw_value_;
		loss_value = loss_value_;
		type = type_;
        use_kernel = use_kernel_;

        // Random generator


        dist = boost::random::uniform_int_distribution<>(1,13);


		/// Number of actions
		n_actions = 2;
		n_states = (21 - 12 + 1) * (10 - 1 + 1) * 2 + 1;
		terminal_state = n_states - 1;
		n_features = 4;
		//n_features *= 2;

		vi netw(2+num_hidden,n_features);
		netw[netw.size()-1] = 1;

		if(type == App_Linear){
			n_features = 4;
			if(mode == IRL_CRPB){
				n_features = 4;
			}
            n_features = 14;
			app = new LinearApproximator(n_features);
			qapp = new LinearApproximator(n_features*2);
			MultipleNN = false;
		}

		else if(type == App_NN){

			n_features = 3;
			netw = vi(2+num_hidden,n_features);
			netw[netw.size()-1] = 1;
			for(int i = 0; i<num_hidden; i++){

				netw[i+1] = layerlist[i];
			}
			//netw.resize(3, n_features);
			//netw=vi(3,n_features);
			//netw[1] = 5*2;
			//netw[2] = 1;
			//app = new NNApproximator(netw);
			//app = new LinearApproximator(n_features);

			if(mode == IRL_CRPB){
			app = new FNNApproximator(netw, ACT_TANH);
			//qapp = new NNApproximator(netw,2);
			qapp= new FNNApproximator(netw, ACT_TANH,2);
			}

			else if(mode == IRL_CPRB){

				app = new FNNApproximator(netw, ACT_TANH);
				qapp = new FNNApproximator(netw,ACT_TANH,2);
			}
			MultipleNN = true;

		}

		//
		//app = new NNApproximator(n_features);




		//qapp = new FNNApproximator(netw, ACT_TANH);
		//qapp = new NNApproximator(n_features + 1);

		//qapp = app;
		vapp = app;
		rapp = app;

        //mdp = getMDP();
		srand48(time(NULL));
		srand(time(NULL));

		setRandomSeed(time(NULL));

		MersenneTwisterRNG *mersenne_twister = new MersenneTwisterRNG();
		rng = (RandomNumberGenerator*) mersenne_twister;
		rng->manualSeed(time(NULL));
		reset();
		std::cout << " Cool " << std::endl;
		compute_basis(n_features);
	}

	virtual int get_n_actions(int s = 0) {
			assert(s<n_states);

			if(s == 200){
				return 0;
			}
			return n_actions;
      }

	virtual DiscreteMDP* getMDP() {
		/// Initialization of the MDP

        std::cerr<<"145 AppBlackjack.h Get MDP is  currently not working"<<std::endl;

        exit(-1);

		mdp = new DiscreteMDP(n_states, n_actions);

		/// Initialization of the reward

		for (int i = 0; i < n_states; i++) {
			for (int k = 0; k < 2; k++)
				mdp->setFixedReward(i, k, 0);
		}

		/// Setting the reward and the MDP  state transition probability kernel

		double pr = 1.0 / 13.0;

		mdp->setTransitionProbability(terminal_state, 0, terminal_state, 1.0);
		mdp->setTransitionProbability(terminal_state, 1, terminal_state, 1.0);

		for (uint pci = 12; pci < 22; pci++) {
			for (uint dci = 1; dci < 11; dci++) {
				for (uint isUsablei = 0; isUsablei < 2; isUsablei++) {
					int s = getState(pci, dci, isUsablei);

					for (uint cardi = 1; cardi < 11; cardi++) {
						if (cardi == 10)
							pr = 4.0 / 13.0;
						else
							pr = 1.0 / 13.0;

                        /// This is just a workaround
                        /// It is false and should not be used
                        /// Probably the loop on isUsablei should be dropped
                        int tmp_pc = pci + cardi;
                        if(tmp_pc > 21 && isUsablei == 1){
                            tmp_pc -= 10;
                            //isUsablei  = 0;
                        }
                        int s_next = getState(tmp_pc, dci, isUsablei);
						double tmp1 = mdp->getTransitionProbability(s, 0,
								s_next);
						mdp->setTransitionProbability(s, 0, s_next, pr + tmp1);
						mdp->setTransitionProbability(s, 1, terminal_state,
								1.0);

					}
					mdp->setFixedReward(s, 1, 0);
					mdp->setFixedReward(s, 0, 0);

				}

			}
		}

		return mdp;
	}
	;

	/* int getState()
	 {
	 return state;
	 }*/

	void compute_basis(int num = 3){


		if(type == App_Linear){

		v_basis.reserve(n_states+2);

		int v_num = num;
        //if(num>=6){

            //v_num= num-3;
        //}
		VectorXd tmp=VectorXd::Ones(v_num);
		int pc2, dc2, isN2;


		for(int i = 0; i<n_states; i++){

			getTriplet(pc2, dc2, isN2, i);

            //tmp(0) = (pc2-16.5)/9.0;

            //tmp(1) = (dc2-6.11)/9.0;

            //tmp(2) = (isN2-0.5);

            tmp(0) = pc2;
            tmp(1) = dc2;
            tmp(2) = isN2;
            tmp(3) = pc2*pc2;
            tmp(4) = dc2*dc2;
            tmp(5) = isN2*isN2;
            tmp(6) = pc2*dc2;
            tmp(7) = pc2*isN2;
            tmp(8) = dc2 * isN2;
            tmp(9) = pc2*dc2*isN2;
            tmp(10) = pc2*pc2*pc2;
            tmp(11) = dc2*dc2*dc2;
            tmp(12) = isN2*isN2*isN2;
			v_basis.push_back(tmp);


		}
        scale(v_basis,-1,1);

		//DiscreteMDP* mdp = getMDP();
		q_basis.resize(n_states);
		VectorXd tmp2 = VectorXd::Zero(v_num*2);
		for(int i =0 ;i< n_states; i++){

			for(int j = 0; j < v_num ; j++){

				tmp2(j) = v_basis[i](j);
			}




			//tmp2(3)=  0;
			//tmp2(4) = 0;
			//tmp2(5) = 0;
			for(int j=0;j<v_num;j++){

				tmp2(j+v_num)=0;
			}

			q_basis[i].push_back(tmp2);


			for(int j=0;j<v_num;j++){

				tmp2(j+v_num) = tmp2(j);
			}

			//tmp2(3)=tmp2(0);
			//tmp2(4) =tmp2(1);
			//tmp2(5) = tmp2(2);

			for(int j=0;j<v_num;j++){

				tmp2(j) = 0;
			}

			//tmp2(0)=  0;
			//tmp2(1) = 0;
			//tmp2(2) = 0;

			q_basis[i].push_back(tmp2);



			//tmp2(num) = 0;
			//q_basis[i].push_back(tmp2);
			//tmp2(num) = 1;
			//q_basis[i].push_back(tmp2);
			//tmp2(num)=0.5;
			//tmp2(num+1)=1;
			//q_basis[i].push_back(tmp2);
			//tmp2(num+1)=0.5;
			//tmp2(num)=1;
			//q_basis[i].push_back(tmp2);

		}

		/*for(int i =0; i< n_states; i++){

			for(int j =0; j < 2; j++){
				tmp = VectorXd::Zero(num);

				for(int k = 0; k < n_states; k++){

					tmp += mdp->getTransitionProbability(i,j,k)*v_basis[k];
					//tmp += v_basis[k];


				}
				q_basis[i].push_back(tmp);
			}


		}*/
		}

		else if(type == App_NN){


			v_basis.reserve(n_states+2);
			v_basis = std::vector<VectorXd>();
			q_basis =std::vector<std::vector<VectorXd> >(n_states);
			VectorXd tmp=VectorXd::Ones(num);
			int pc2, dc2, isN2;



			for(int i = 0; i<n_states; i++){

				getTriplet(pc2, dc2, isN2, i);

				//tmp(0) = (pc2-12)/10;
				tmp(0) = pc2;
				//tmp(1) = dc2/10;
				tmp(1) = dc2;
				//tmp(2) = isN2;
				tmp(2) = isN2;
				v_basis.push_back(tmp);

				q_basis[i].push_back(tmp);
				q_basis[i].push_back(tmp);


			}







		}

		v_basis_bis = std::vector<VectorXmp>(v_basis.size());
		q_basis_bis = std::vector<std::vector<VectorXmp> >(q_basis.size());


		for(int i=0; i< v_basis.size(); i++){

            v_basis_bis[i] = v_basis[i].cast<mpreal>();

		}

		for(int i=0; i<q_basis.size();i++){


            for(int j=0; j<q_basis[i].size();j++){


                q_basis_bis[i].push_back(q_basis[i][j].cast<mpreal>());
            }
		}


	}

	virtual VectorXd& get_q_basis(int i_ep, int i_ns, DEMO_MODE demo_st, int s,
				int a = -1, int s_next = -1) {

		assert(s_next == -1 && a != -1);
		return q_basis[s][a];
	}

	virtual VectorXd& get_basis(int i_ep, int i_ns, DEMO_MODE demo_st, int s,
			int a = -1, int s_next = -1, bool q_features_available_ = false) {

		if(s_next != -1 && a != -1){

			return v_basis[s_next];
		}

		else if(s_next == -1 && a != -1){

			return q_basis[s][a];




		}

		return v_basis[s];


	}

	virtual VectorXmp& get_q_basis(int s,int a = -1, int s_next = -1) {

		assert(s_next == -1 && a != -1);
		return q_basis_bis[s][a];
	}

	virtual VectorXmp& get_basis(int s,int a = -1, int s_next = -1) {

		if(s_next != -1 && a != -1){

			return v_basis_bis[s_next];
		}

		else if(s_next == -1 && a != -1){

			return q_basis_bis[s][a];




		}

		return v_basis_bis[s];


	}

	/*Matrix get_basis(int s, int a = -1, int s_next = -1) {

		Matrix tmp(3, 1);
		int pc_, dc_, isU_, pc2, dc2, isU2, s2, t, f;
		if (s_next != -1 && a != -1) {

			getTriplet(pc_, dc_, isU_, s_next);
			tmp(0, 0) = pc_;
			tmp(1, 0) = dc_;
			tmp(2, 0) = isU_;
		} else if (s_next == -1 && a != -1) {
			getTriplet(pc2, dc2, isU2, s);
			tmp(0, 0) = 0;
			tmp(1, 0) = 0;
			tmp(2, 0) = 0;
			f = 1;
			for (int i = 1; i < 11; i++) {
				t = i;
				if (i == 1 && pc2 + i <= 11)
					t += 10;
				if (i == 10)
					f = 4;
				s2 = getState(pc2 + t, dc2, isU2);
				getTriplet(pc_, dc_, isU_, s2);
				tmp(0, 0) += f * pc_;
				tmp(1, 0) += f * dc_;
				tmp(2, 0) += f * isU_;

			}

			for (int i = 0; i < 3; i++) {
				tmp(i, 0) /= 13.0;
			}

		} else {

			getTriplet(pc_, dc_, isU_, s);
			tmp(0, 0) = pc_;
			tmp(0, 1) = dc_;
			tmp(0, 2) = isU_;

		}

		return tmp;

	}*/

	void reset() {
		/// The player has not a Blackjack or Natural
		isNatural = false;

        // The dealer has not a usable ace
        dealer_usable = false;

		/// Dealing the first card of the Dealer

        //first_value = std::min((int) rng->discrete_uniform(13) + 1, 10);
        first_value = std::min(dist(gen),10);

		/// Dealing the second card of the dealer
        //second_value = std::min((int) rng->discrete_uniform(13) + 1, 10);
        second_value = std::min(dist(gen),10);
		//second_value=std::min(rand()%13+1,10);

		/// Dealing cards to the player until he has 12 or more
		int state_ini = 0;
		int isU = 0;
		int num = 0;
		while (state_ini < 12) {
            //int tmp = std::min((int) rng->discrete_uniform(13) + 1, 10);
			//int tmp=std::min(rand()%13+1,10);
            int tmp = std::min(dist(gen),10);

			state_ini += tmp;
			num++;

			if (tmp == 1 && state_ini < 12) {
				state_ini += 10;
				isU = 1;
			}

		}

		/// If the player has only a Face card and an Ace then he has a natural

		if (isU == 1 && state_ini == 21 && num == 2)
			isNatural = true;

		/// If the dealer first two cards contains an Ace then Add 10 to his second_value

        if ((first_value == 1) || (second_value == 1)){
            second_value += 10;
            dealer_usable = true;
        }


		/// Initialisation of the initial state
		state = (state_ini - 12) * 10 + first_value - 1 + 100 * (isU);

		/// Getting the triplet pc dc and isUsable for the initial state
		getTriplet(pc, dc, isUsable, state);

		assert(pc==state_ini && dc==first_value && isUsable==isU);

		/// Setting the reward to zero
		reward = 0.0;
	}


	/**
	 * Performing  action from state to a new state
	 */
	 bool act(int action, bool autom = true, int s2 = -1) {
		//std::cout<<"yes\t";

		int state1;
		state1 = state;

		/// Generating the resulting state corresponding to the effect of action on the current state

		//state=mdp->generateState(state1,action);

		/// Hitting
		if (action == 0) {
            //int tmp1 = std::min((int) rng->discrete_uniform(13) + 1, 10);
			//int tmp1=std::min(rand()%13+1,10);
            int tmp1 = std::min(dist(gen),10);
            pc += tmp1;

            // Checking if player sum is greater than 21 and the player
            // has a usable ace
            if(isUsable == 1 && pc > 21){

                // With this we can not have a blackjack nor a usable ace
                isUsable = 0;
                isNatural = false;
                pc -= 10;
            }
            state = getState(pc, dc, isUsable);
            //state = getState(pc + tmp1, dc, isUsable);
            //pc += tmp1;
		}

		/// Standing
		if (action == 1)
			state = terminal_state;

		/// If action is to stand or state is terminal_state computing the reward for the player
		if (state == terminal_state || action == 1) {
			//dc+=second_value;

			/// Updating dealer total point
			dc = first_value + second_value;
			reward = calculReward();

			return false;
		} else {
			reward = 0.0;
			return true;
		}
	}

	//void Show();

	int getState1() const {
		return state;
	}

	/**
	 * Computing the reward of the player given
	 * the state triplet and the value of isNatural
	 * -1 for a loss, 1 for a win, 0.0 for a draw
	 *
	 */
    double calculReward() {

        /// Checking if player sum is greater than 21 and the player
        /// has a usable ace

//        if(isUsable == 1 && pc > 21){
//            pc -= 10;
//            // With this we can not have a blackjack
//            isNatural = false;
//        }




		/// Player bust
        if (pc > 21){



            return loss_value;

        }



		/// Player got a natural
		if (isNatural && pc == 21) {
			if (dc == 21)
				return draw_value;
			else
				return win_value +0.5;

		}

		/// Including natural win for the dealer ???

         if(dc==21)
            return loss_value;



		/// The dealer is playing
		int dhv = dc;


		while (dhv < 17) {
            //int j = std::min((int) rng->discrete_uniform(13) + 1, 10);
			// int j=std::min(rand()%13+1,10);
            int j = std::min(dist(gen),10);
			dhv += j;
            if (j == 1 && dhv < 12){

                dhv += 10;
                dealer_usable = true;
            }

            if(dealer_usable && dhv > 21){

                dealer_usable = false;
                dhv -= 10;
            }






		}

		/// Dealer bust
		if (dhv > 21)
			return win_value;

		/// Tie

		if (dhv == pc)
			return draw_value;

		/// Player win
		if (dhv < pc)
			return win_value;

		/// Player loose
		if (dhv > pc)
			return loss_value;
		std::cout << " Error should not come here ";
		exit(-1);
		return 0;
	}


	/**
	 * Given the state triplet pc, dc and isUsable
	 * state returns a unique identification number
	 */
	int getState(int pc_, int dc_, int isUsable_) const {
		assert(pc_>=12 && dc_>=1);
		int s = (pc_ - 12) * 10 + (dc_ - 1);
        assert((isUsable_==1 && pc_ <= 21) || isUsable_ == 0);
		if (isUsable_ == 1) {
			s += 100;
			s = std::min(s, terminal_state);
		}

		else {
			if (s >= 100)
				s = terminal_state;
		}

		return s;
	}


	/**
	 *  Compute the corresponding triplet pc,dc,isUsable
	 *  from the current state
	 */

	void getTriplet(int& pc_, int& dc_, int& isUsable_,
			const int& state_2) const {
		isUsable_ = 0;
		int state_ = state_2;
		if (state_ >= 100) {
			state_ -= 100;
			isUsable_ = 1;
		}

		pc_ = state_ / 10;
		dc_ = state_ % 10;
		pc_ += 12;
		dc_ += 1;
	}


	/**
	 * Return the name of the environment
	 */

	virtual std::string Name() {
		return "Blackjack";
	}

	int getState() {
		return state;
	}
	virtual ~AppBlackjack() {
		//dtor
        //delete mdp;
	}

public:
	double win_value;
	double draw_value;
	double loss_value;
	int terminal_state;
	int pc, dc, isUsable, second_value, first_value;
	/// True is the first two cards of the player is a Face and an Ace
	bool isNatural;
    bool dealer_usable;
	RandomNumberGenerator *rng;

    boost::random::uniform_int_distribution<> dist;

private:
};

#endif /* APPBLACKJACK_H_ */
