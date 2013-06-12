//============================================================================
// Name        : RL.cpp
// Author      : TOSSOU Aristide
// Version     :
// Copyright   : Linux
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <fstream>

#include "ValueIteration.h"

#include "MyTypes.h"
#include "AppEnvironment.h"
#include "IRL.h"
#include "AppTTT.h"
#include "AppBlackjack.h"
#include "VIAMG.h"
//#include "Optimize.h"
#include "MatlabWrapper.h"
#include "LRP.h"
//using namespace std;

int n_run = 10; // Number of time we will run each environment
int n_step = 35000000; // This is the max numbers of step for all possible episodes

int n_episodes = 100; // Number max of episode
int limit_ep = -1;  // This is the number of episodes before observing

int loss_point = -1;  // Point when loss
int draw_point = 0;
int win_point = 1;

int n_episodes_play = 200000; /// Number of episodes to test performance

int testOpp = 0;   /// Opponent type for testing

bool firstGreedy = false;

/// Policy Performance routines for approximate environment

void PolicyPerformance_app(AppTTT& env, VectorXd& policy, double& point,
        int& num, int &draw, int &loss, vii& all_best_action, bool index = false, bool isV = false) {

	point = 0;
	num = 0;

	int episode = -1;
	bool action_ok = false;
	draw = 0;
	loss = 0;

	for (int step = 0; step < n_step; ++step) {
		if (!action_ok) {
            if (episode >= 0) {
				double reward = env.get_reward();
				num++;
				if (reward == loss_point)
					loss++;
				if (reward == draw_point)
					draw++;

			}
			episode++;
            if (n_episodes_play >= 0 && episode >= n_episodes_play) {

				break;
			} else {

				env.reset();
				//policy.Reset();

				action_ok = true;

			}
		}

		int state = env.get_state();

		int action;

		//policy.Observe(reward, state);
		//action =policy.SelectAction();


        action = env.policy_act(policy, state);


		if (index) {
			//std::cerr<<"dep\t State = "<<state<<"\t"<<"action = "<<action<<"\t";
			//std::cerr<<environment.transition.size()<<"\t";
			action = env.get_state_index(state, action);
			//std::cerr<<"bon?\t";
		}

		if(testOpp == 0){
			action_ok = env.act(action, true, -1);
		}

		else if(testOpp == 1){

			action_ok = env.act(action, true , 2);
		}

        else if(testOpp == 2){

            // against  random optimal player
            // Playing optimal action at random between the best ones

            int s2 = env.transition[state][action];

            //if(env.term[action])
            if(env.term[s2])
                action_ok = env.act(action);
            else
                action_ok = env.act(action,false,all_best_action[s2][rand()%all_best_action[s2].size()]);


                //action_ok = env.act(action,false,all_best_action[action][rand()%all_best_action[action].size()]);
        }



		//action_ok = env.act(action);

		double reward1 = env.get_reward();
		point += reward1;

	}
}


void PolicyPerformance_app(AppTTT& env, std::vector<int>& policy,
        double& point, int& num, int &draw, int &loss,vii& all_best_action, bool index = false,
        bool isOpt = false) {

	point = 0;
	num = 0;
    std::cerr<<"Testing using vector"<<std::endl;

	int episode = -1;
	bool action_ok = false;
	draw = 0;
	loss = 0;

	for (int step = 0; step < n_step; ++step) {
		if (!action_ok) {
            if (episode >= 0) {
				double reward = env.get_reward();
				num++;
				if (reward == loss_point)
					loss++;
				if (reward == draw_point)
					draw++;

			}
			episode++;
            if (n_episodes_play >= 0 && episode >= n_episodes_play) {

				break;
			} else {

				env.reset();
				//policy.Reset();

				action_ok = true;

			}
		}

		int state = env.get_state();

		int action;

		//policy.Observe(reward, state);
		//action =policy.SelectAction();

        if (!isOpt) {
			action = policy[state];
        }
        else{

        if(firstGreedy){
            action = policy[state];
        }
        else{

            action = all_best_action[state][rand()%all_best_action[state].size()];
        }
        }

		if (index) {
			//std::cerr<<"dep\t State = "<<state<<"\t"<<"action = "<<action<<"\t";
			//std::cerr<<environment.transition.size()<<"\t";
			action = env.get_state_index(state, action);
			//std::cerr<<"bon?\t";
		}

		if(testOpp == 0){
					action_ok = env.act(action, true, -1);
				}

				else if(testOpp == 1){

					action_ok = env.act(action, true , 2);
				}

        else if(testOpp == 2){

            // against  random optimal player
            // Playing optimal action at random between the best ones

            int s2 = env.transition[state][action];
            //if(env.term[action])
            if(env.term[s2])
                action_ok = env.act(action);
            else
                action_ok = env.act(action,false,all_best_action[s2][rand()%all_best_action[s2].size()]);


                //action_ok = env.act(action,false,all_best_action[action][rand()%all_best_action[action].size()]);
        }

		//action_ok = env.act(action);

		double reward1 = env.get_reward();
		point += reward1;

	}

}


/// This function takes a deterministic policy where the action are always
/// number from 0 to n_actions. It does not do any modification if the
/// expected action by the environnement is different. The environment
/// should take care of that

void PolicyPerformance(AppEnvironment& env, std::vector<int>& policy,
        double& point, int& num, int &draw, int &loss) {

    point = 0;
    num = 0;

    int episode = -1;
    bool action_ok = false;
    draw = 0;
    loss = 0;

    for (int step = 0; step < n_step; ++step) {
        if (!action_ok) {
            if (episode >= 0) {
                double reward = env.get_reward();
                num++;
                if (reward == loss_point)
                    loss++;
                if (reward == draw_point)
                    draw++;

            }
            episode++;
            if (n_episodes_play >= 0 && episode >= n_episodes_play) {

                break;
            } else {

                env.reset();


                action_ok = true;

            }
        }

        int state = env.get_state();

        int action;

        action = policy[state];

        if(testOpp == 0){
                    action_ok = env.act(action, true, -1);
                }

                else if(testOpp == 1){

                    action_ok = env.act(action, true , 2);
                }



        double reward1 = env.get_reward();
        point += reward1;

    }

}



void save_stats(std::string root, int s_dev, int num, double point, int draw, int win, int loss, double time_algo, std::vector<int>& greedy_pol){

    // Saving deviation, points, number of loss, draw, win, policy
    std::ofstream dev_out((root+"dev").c_str());
    dev_out<<s_dev<<std::endl;
    std::ofstream num_out((root+"numtest").c_str());
    num_out<<num<<std::endl;
    std::ofstream points_out((root+"points").c_str());
    points_out<<point<<std::endl;
    std::ofstream draw_out((root+"draw").c_str());
    draw_out<<draw<<std::endl;
    std::ofstream win_out((root+"win").c_str());
    win_out<<win<<std::endl;
    std::ofstream loss_out((root+"loss").c_str());
    loss_out<<loss<<std::endl;
    std::ofstream time_out((root+"time").c_str());
    time_out<<time_algo<<std::endl;
    std::ofstream pol_out((root+"policy").c_str());
    for(int s=0; s<greedy_pol.size();s++){

        pol_out<<greedy_pol[s]<<"\t";
    }
    dev_out.close();
    num_out.close();
    points_out.close();
    draw_out.close();
    win_out.close();
    loss_out.close();
    time_out.close();
    pol_out.close();
}

int main(int argc, char **argv) {


	srand48(time(NULL));
	srand(time(NULL));
	std::cout << "Starting test program" << std::endl;

	//int ep_run[10]={2000,1000,1500,500,250,200,150,100,75,50};
	//int ep_run[10]={2000,10,50,500,250,200,150,100,75,50};
	n_run = 1;

	std::string arg = "";
	std::string root="";
    std::string demo_file="";

	std::string game="blackjack";
	std::string algo="LRPB";
	std::string optimiser="GSS";
	OPT optimisateur =OPT_GSS;
	int max_time = 720;
	int freq = 200;
	int n_iterations = 20000;
    int freqForOneRun=1;
	bool MultipleNN = true;
	MultipleNN = true;
	bool useMean = false;

	int secondOpp = 0;
	int reg = 2;
    bool use_kernel = false;
    double kernel_sig = 1;
    int max_points = INT_MAX;
    int q_features_type = 2;

	int num_hidden = 1;

	int randomPIdemo = 0;
	int num_PIdemo = n_episodes_play;

    bool opt_stats = false;

	vi layerlist(num_hidden,3);
	vi runlist(n_run,200);

	int rem_arg= argc;
	int actu_arg = 1;
	std::string cur_opt="";

	std::string help="game : To specify the game to test, possible value are blackjack, tictactoe and st Default : blackjack\n\n"
			"algo : To specify the algorithm to use possible values are LRPB, LPRB, NNPRB and NNRPB Default : LRPB\n\n"
			"opt : To specify the optimisation search algorithm to use. Possibles values are GSS ( Generating"
			"set search, SOLIS, MC (Simple monte-carlo search), LRS( Localised random search). Default : GSS \n\n"
			"max_time : to specify the maximum time for each run of the algorithm in second. Default : 720 \n\n"
			"n_iterations : Number of iterations of the optimisation search. Default : 20000 \n\n"
			"freq :  after freq iterations the optimisation search is reset. Default : 200 \n\n"
			"n_run : Number of run of the algorithm on the specified game. Default : 1 \n\n"
			"runlist: This is a list of n_run numbers separated by space, each number indicate"
			"the number of episodes of the demonstration of the experts. Default : 200\n\n"
			"freqForOneRun : This is the number of times each run will be executed. Default : 10 \n\n"
			"numtestepisodes : This is the number of episodes for the test of the algorithm. Default : 200000\n\n"
			"root : This is a directory where the statistics will be saved. It should be terminated by /\n\n"
			"mean : boolean (true or false) indicates if we should return the policy with the maximum likelihood"
			"or the weighted mean of the reward (for only the 50 most probable) Default : false\n\n"
			"numhiddenlayer : for algorithm using neural network this is the number of hidden layers. Default : 1\n\n"
			"layerlist : This is a list of numhiddenlayer elements, each represents the number of neuron in each hidden layer. Default : 3\n\n"
			"reg : for LRPB the LSPI algorithm is used and we need inverse of a matrix A. This is the regularisation term to initialize A"
			"so that A=reg*I. Default : 2\n"
			"firstgreedy : For tictactoe this indicates whether or not the first player takes random optimal action or greedy"
			"optimal action (for the experts demonstrations), Possible values are 0 (random), 1 (greedy). Default : 0\n\n"
			"secondOpp : For tictactoe this indicates the type of the second player for the experts demonstrations. Possible"
			"values are 0 (random player) 1 (semi random player) 2 (random optimal player). Default : 0\n\n"
			"randompidemo: Boolean (0 or 1) This indicates whether or not the demonstration for LSPI or FittedQ is the same as the opti"
			"mal demonstrations. 0 means it is the same 1 means it is different. If it is different the lspi demonstration"
			"is the games between two random players. Default: 1\n\n"
			"numpidemo If randompidemo is 1 this indicates the number of episodes of the lspi demonstration. Default : 2000\n\n"
            "testopp : For tictactoe indicates the type of the second player for the tests. Possible values are 0 (random) 1 (semi random) 2 (random optimal)"
			"Default : 0\n\n"
			"help : to print the help\n\n";


	while(actu_arg < rem_arg){

		cur_opt = argv[actu_arg];

		if(cur_opt == "help"){

					std::cout<<help<<std::endl;
					exit(0);
				}
		actu_arg++;



		if(actu_arg >= rem_arg)
			continue;



		if(cur_opt == "game"){
			game = argv[actu_arg];
			arg = game;
            //actu_arg++;

		}

		else if(cur_opt == "algo"){

			algo = argv[actu_arg];
            //actu_arg++;
		}

		else if(cur_opt == "opt" ){

			optimiser = argv[actu_arg];

            //actu_arg++;

		}
		else if(cur_opt == "max_time" ){

			max_time = atoi(argv[actu_arg]);

            //actu_arg++;

		}
		else if(cur_opt == "freq" ){

			freq = atoi(argv[actu_arg]);

            //actu_arg++;

		}
		else if(cur_opt == "n_iterations" ){

			n_iterations = atoi(argv[actu_arg]);

            //actu_arg++;

		}
		else if(cur_opt == "freqForOneRun" ){

			freqForOneRun = atoi(argv[actu_arg]);

            //actu_arg++;

		}

		else if (cur_opt == "nrun" ){


			n_run = atoi(argv[actu_arg]);

			runlist = vi(n_run,200);

            //actu_arg++;
		}
		else if (cur_opt == "numtestepisodes" ){

			n_episodes_play = atoi(argv[actu_arg]);




            //actu_arg++;
		}
		else if (cur_opt == "root"){

			root=argv[actu_arg];
            if(root[root.size()-1] != '/'){

                std::cerr<<"root should be a directory, you give a file, I will consider the corresponding directory"<<std::endl;
                root = root+"/";
            }




            //actu_arg++;
		}

        else if(cur_opt == "demo_file"){

            demo_file = argv[actu_arg];
            assert(demo_file.size() > 0);
            if(demo_file[demo_file.size()-1] == '/'){

                std::cerr<<"Demo File should be a file, you give a directory, I will consider the corresponding file"<<std::endl;
                demo_file = demo_file.substr(0,demo_file.size()-1);
            }


        }
		else if (cur_opt == "mean" ){

			std::string tmp =argv[actu_arg];

			if(tmp=="true"){
				useMean = true;
			}
			else if( tmp=="false"){

				useMean = false;

			}

            //actu_arg++;

		}

		else if( cur_opt == "numhiddenlayer"){

			num_hidden = atoi(argv[actu_arg]);
			if(num_hidden > 0)
			layerlist = vi(num_hidden, 3);
		}

		else if (cur_opt == "layerlist"){

			for(int i = 0; i < num_hidden; i++){

				if(actu_arg < argc){

					layerlist[i] = atoi(argv[actu_arg]);
					actu_arg++;
				}
			}

            // Her actu_arg point to the next element in the list
            // We don't need that
            actu_arg--;
		}

		else if (cur_opt == "runlist"){

			for(int i = 0; i < n_run ; i++){

				if(actu_arg < argc){

					runlist[i] = atoi(argv[actu_arg]);

					actu_arg++;

				}

			}

            // Her actu_arg point to the next element in the list
            // We don't need that
            actu_arg--;
		}

		else if(cur_opt == "reg"){

			reg = atoi(argv[actu_arg]);
            //actu_arg++;
		}

		else if(cur_opt == "firstgreedy"){

			std::string tmp = argv[actu_arg];
			if(tmp == "true"){
				firstGreedy = true;
			}
			else if(tmp == "false"){

				firstGreedy = false;
			}

            //actu_arg++;
		}

		else if(cur_opt == "secondopp"){

			secondOpp = atoi(argv[actu_arg]);

            //actu_arg++;

		}

		else if( cur_opt == "randompidemo"){

			randomPIdemo = atoi(argv[actu_arg]);
            //actu_arg++;
		}

		else if(cur_opt == "numpidemo"){

			num_PIdemo = atoi(argv[actu_arg]);
            //actu_arg++;
		}

		else if(cur_opt == "testopp"){

			testOpp = atoi(argv[actu_arg]);
            //actu_arg++;
		}

        else if(cur_opt == "use_kernel"){

            std::string tmp = argv[actu_arg];
            if(tmp =="true"){
                use_kernel = true;
            }
            else{
                use_kernel = false;
            }

            //actu_arg++;
        }

        else if(cur_opt=="opt_stats"){
            std::string tmp = argv[actu_arg];
            if(tmp=="true"){
                opt_stats = true;
            }
            else {
                opt_stats = false;
            }


        }

        else if(cur_opt == "kernel_sig"){

            kernel_sig = atof(argv[actu_arg]);
            //actu_arg++;
        }

        else if(cur_opt == "q_features_type"){
            q_features_type = atoi(argv[actu_arg]);
            //actu_arg++;
        }

        else if(cur_opt == "max_kernel_points"){

            max_points = atoi(argv[actu_arg]);
            //actu_arg++;
        }


        // Now Let's point actu_arg to the next element
        // This should be the last thing

        actu_arg++;




	}

	/*
	 * First argument:  Game (Blackjack , ...
	 * Second:   algorithm  ( LRPB, LPRB...
	 * Thrid:   Optimiser ( MC, LRS, NLOPT)
	 * Fourth:  Max_time
	 * Fifth  freq
	 * Sixth:  N_iterations
	 * Seven:  FreqForOneRun
	 * Eight:  n_run
	 * Nine:   Using mean or not: (mean, nomean)
	 * Ten:    Root of the directory for the output
	 * Eleven: Number of games to test the performance
	 *
	 */
//	if (argc >= 2){
//		arg = argv[1];
//		game = argv[1];
//		algo = argv[2];
//		optimiser = argv[3];
//
//		if(argc>=5){
//			max_time = atoi(argv[4]);
//		}
//
//		if(argc>=6){
//
//			freq = atoi(argv[5]);
//
//		}
//
//		if(argc>=7){
//
//			n_iterations = atoi(argv[6]);
//		}
//
//		if(argc>=8){
//
//			freqForOneRun = atoi(argv[7]);
//		}
//
//		if(argc>=9){
//
//			n_run = atoi(argv[8]);
//		}
//
//		if(argc>=10){
//
//			std::string tmp =argv[9];
//
//			if(tmp=="mean"){
//				useMean = true;
//			}
//			else if( tmp=="nomean"){
//
//				useMean = false;
//
//			}
//		}
//
//		if(argc >=11){
//
//			root=argv[10];
//		}
//
//		if(argc >= 12){
//
//			n_episodes_play = atoi(argv[11]);
//		}
//
//
//
//	}


	if (arg == "st") {

		//std::cout << "Computing ST Based environemnt";
		//exec();
		//exit(0);
	}

	bool toggleTTT = false;

	bool toggleBL=  false;

	ALG alg = ALG_LPRB;
	IRL_MODE mode = IRL_CPRB;
	App approximateur = App_Linear;

	if(game == "blackjack"){

		toggleBL = true;
		alg = ALG_LPRB;
	}
	else if(game == "tictactoe"){

		toggleTTT = true;
		alg = ALG_LRPB;
	}

	if(algo=="LRPB"){
		alg = ALG_LRPB;
	}
	else if(algo == "LPRB"){

		alg = ALG_LPRB;
	}

	else if( algo == "NNRPB"){

		alg = ALG_NNRPB;
	}

	else if( algo == "NNPRB"){

		alg = ALG_NNPRB;
	}
    else if(algo == "LRP"){

        alg =ALG_LRP;
	}

    else if(algo == "LPR"){
        alg = ALG_LPR;
    }
    else if(algo == "Optimal"){
        alg =ALG_OPT;
    }
    else if(algo == "Manual"){
        alg = ALG_MANUAL;
    }

	if(optimiser == "GSS"){
		optimisateur =OPT_GSS;

	}
	else if(optimiser == "LRS"){

		optimisateur =OPT_LRS;
	}

	else if(optimiser == "MC"){

		optimisateur =OPT_MC;
	}
	else if(optimiser == "NLOPT"){
		optimisateur = OPT_NLOPT;
	}

	else if (optimiser == "SOLIS"){

		optimisateur = OPT_SOLIS;
	}

	else if(optimiser == "GD"){

		optimisateur = OPT_GD;
	}




	/// Defining RewardPolicyBelief or PolicyRewardBelief
	if(alg==ALG_LRPB || alg==ALG_NNRPB){

		mode = IRL_CRPB;
	}

	else if(alg==ALG_LPRB || alg== ALG_NNPRB){

		mode = IRL_CPRB;
	}


	/// Defining linear or neural network

	if(alg==ALG_LRPB || alg==ALG_LPRB){

		 approximateur = App_Linear;
	}
	else if(alg == ALG_NNPRB || alg == ALG_NNRPB){

		approximateur = App_NN;
	}

    if(alg == ALG_LRP || alg == ALG_LPR){
        approximateur = App_Linear;
        optimisateur = OPT_LRP;

	}



	if(game == "st"){

		exec(approximateur,  mode ,
				max_time , n_iterations,  freq ,
                optimisateur ,useMean, num_hidden, layerlist, reg ,root,  use_kernel, kernel_sig, max_points, q_features_type,alg );


		return 0;
	}








	/// Creating statistics

	int stat_num = 2;
	int s_dev = 0;
	int s_dev_tot = 0;

	//std::vector<Statistics> stat(stat_num);

	std::vector<std::vector<Statistics> > stat(n_run, std::vector<Statistics>(stat_num));

	for(int run =0; run < n_run; run++){
	for (int i = 0; i < stat_num; i++) {

		stat[run][i].n_games = 0;
		stat[run][i].num_draw = 0;
		stat[run][i].num_loss = 0;
		stat[run][i].reward = 0;
		stat[run][i].name="";
	}
	}
	vi results(n_run,0);
	vi deviations(n_run,0);

	//std::ofstream policyout("/home/yedtoss/workspace/RL3/data/policyNNPRBTTT.txt");
	std::ofstream policyout((root+"policy.txt").c_str());

	for (int run = 0; run < n_run; run++) {


		for(int fr = 0; fr < freqForOneRun; fr++){

		//n_episodes = ep_run[run];
			n_episodes = runlist[run];

		bool action_ok; // allow us to know if an episode is finished or not
		bool start1;   // we will only start observing if start1 is true
		int episode = 1; // To keep the current number of episodes
        int num, draw, loss, win;
        double time_algo = 0;
		double point;

		//AppBlackjack* env_bl = NULL;
		//env_bl=new AppBlackjack();

		/// TTT Environment

		/// Computing optimal policy



		while (toggleTTT) {

			AppTTT env_ttt(approximateur);
			MG *mdp_ttt = env_ttt.getMDP();
            int n_states = env_ttt.get_n_states();
			std::vector<int> rew_ttt = env_ttt.getRew();
			VIAMG alg_ttt(mdp_ttt, rew_ttt, 1);
			alg_ttt.ComputeStateValues(0.00000000002, 20); /// End of optimal policy
            std::vector<int> greedy_pol;
            vi opt_pol(n_states,-1);
            VectorXd irl_policy;

            for(int s=0; s<n_states; s++){

                if(env_ttt.term[s] == 0){
                    opt_pol[s] = env_ttt.get_index_state(s,alg_ttt.policy[s]);
                }


            }

            assert(n_states == (int)alg_ttt.policy.size());


            vii all_best_action(n_states);

            for(int s=0; s<n_states; s++){

                bool hasImmediateWinningState = false;

                for(int j=0; j<(int)env_ttt.transition[s].size();j++){

                    if(std::abs(alg_ttt.V(alg_ttt.policy[s])-alg_ttt.V(env_ttt.transition[s][j]) ) < 1e-9){

                        if(env_ttt.term[env_ttt.transition[s][j]]){

                            if(!hasImmediateWinningState){
                                hasImmediateWinningState = true;
                                all_best_action[s].clear();
                            }
                            all_best_action[s].push_back(j);


                        }

                        if(!hasImmediateWinningState){

                        all_best_action[s].push_back(j);
                        }


					}
				}
			}


            vi virtual_state;
            vi inv_state(n_states,-1);

            for(int s=0; s <n_states; s++){
                if(env_ttt.player[s]==1 && env_ttt.term[s]==0){
                    continue;
                }
                inv_state[s] = virtual_state.size();
                virtual_state.push_back(s);


            }

            int v_n_states = virtual_state.size();
            int v_term_state = v_n_states;
			/// Getting the Demonstration
            Demonstration demo_ttt, pi_demo_ttt,v_demo_ttt;

             if(demo_file ==""){

			std::cout << " Observing TTT agent Demonstration" << std::endl;

			action_ok = false; //
			episode = -1;
			start1 = false;

			for (int step = 0; step < n_step; step++) {

                // Regle now
				if (!action_ok) {


					episode++;

                    if (episode >= n_episodes) {

						break;
					}

					else {

						env_ttt.reset();
						action_ok = true;
                        if(episode >= limit_ep){
                            demo_ttt.NewEpisode();
                            v_demo_ttt.NewEpisode();
                        }
					}
				}

				int state = env_ttt.get_state();
				//real reward = env_ttt.getReward();
				start1 = true;

				int action = 0;


				if(firstGreedy){
				// Taking a greedy optimal action
                //action = alg_ttt.policy[state];
                    action = opt_pol[state];
				}
				else{
                // Taking a random action between the best ones
				action = all_best_action[state][rand()%all_best_action[state].size()];
				}
                //int action_index = env_ttt.get_index_state(state, action);
                int action_index = action;


				if(secondOpp ==0 ){
                // against random player
				action_ok = env_ttt.act(action,true,-1);
				}

				else if(secondOpp == 1){
				// against semi random player
				action_ok = env_ttt.act(action,true,2);
				}



				else if(secondOpp == 2){
				// against  random optimal player
				// Playing optimal action at random between the best ones

                    int s2 = env_ttt.transition[state][action];

                //if(env_ttt.term[action])
                  if(env_ttt.term[s2])
					action_ok = env_ttt.act(action);
				else


                    //action_ok = env_ttt.act(action,false,all_best_action[action][rand()%all_best_action[action].size()]);

                  action_ok = env_ttt.act(action,false,all_best_action[s2][rand()%all_best_action[s2].size()]);

				}
				int next_state = env_ttt.get_state();
				//reward = env_ttt.getReward();

                if (episode >= limit_ep) {


					demo_ttt.Observe(state, action_index, next_state);
                    v_demo_ttt.Observe(inv_state[state], action_index, inv_state[next_state]);

				}

			}



			if(randomPIdemo){



				action_ok = false; //
							episode = -1;
							start1 = false;

							for (int step = 0; step < n_step; step++) {

								if (!action_ok) {


									episode++;

                                    if (episode >= num_PIdemo) {

										break;
									}

									else {

										env_ttt.reset();
										action_ok = true;
                                        if(episode >= limit_ep){
                                            pi_demo_ttt.NewEpisode();
                                        }
									}
								}

								int state = env_ttt.get_state();
								//real reward = env_ttt.getReward();
								start1 = true;

								int action = 0;

								if(env_ttt.get_n_actions(state) > 0)
                                //action = env_ttt.get_state_index(state,rand()%env_ttt.get_n_actions(state));
                                    action = rand()%env_ttt.get_n_actions(state);
                                //int action_index = env_ttt.get_index_state(state, action);
                                int action_index = action;

								action_ok = env_ttt.act(action,true,-1);



								int next_state = env_ttt.get_state();
								//reward = env_ttt.getReward();

                                if (episode >= limit_ep) {

                                    //start1 = false;
									pi_demo_ttt.Observe(state, action_index, next_state);

								}

							}


			}

			else {

				pi_demo_ttt = demo_ttt;



			}
             }

             // a demonstration file has been given
             else{

                 std::cerr<<"We will load a pre computed demonstration, randompidemo does not work with this mode currently"<<std::endl;

                 demo_ttt = load_demo(demo_file);

                 // Hack we should remove randompidemo later
                 pi_demo_ttt = demo_ttt;

                 //demo_in.close();


             }



             // Computing performance of optimal policy only when required
             if(alg == ALG_OPT || (demo_file == "" && alg != ALG_MANUAL)){


			std::cout << " Evaluating the performance of Optimal Policy on TTT"
					<< std::endl;

            PolicyPerformance_app(env_ttt, opt_pol, point, num, draw,
                    loss,all_best_action, false, true);

			std::cout << point << " Points after " << num << " Plays"
					<< " with "
							"" << loss << " Games lost and " << draw << " Draw "
					<< std::endl;

			stat[run][0].reward += point;
			//stat[run][0].n_games += num;
			stat[run][0].num_draw += draw;
			stat[run][0].num_loss += loss;
			stat[run][0].name = "OptimalTTT";
			stat[run][0].n_games += 0;
            time_algo = 0;
            win = num-draw-loss;

            s_dev = 0;
            // Saving statistics for optimal policy
            save_stats(root,s_dev,num,point,draw,win,loss,time_algo,opt_pol);

            // Save demo for LRP/LPR
            save_demo(demo_ttt,root+"demo",0);

            // Save demo for MaxEnt/FIRL
            assert(env_ttt.term[659]==1);
            save_demo(demo_ttt,root+"demooth",1,9,659,0);

            // Saving virtual demo
            save_demo(v_demo_ttt,root+"v_demo",0);

            save_demo(v_demo_ttt,root+"v_demooth",1,9,inv_state[659],0);

             }


             // If we are only testing the optimal algo
             // we don't need to go further
             if(alg == ALG_OPT && opt_stats){

                 s_dev = 0;
                 //greedy_pol = alg_ttt.policy;

                 // Saving statistics for optimal policy
                 //save_stats(root,s_dev,num,point,draw,win,loss,time_algo,opt_pol);


                 //std::ofstream demout((root+"demo").c_str());
                 //std::ofstream othout((root+"demooth").c_str());
                 std::ofstream featout((root+"feat").c_str());
                 std::ofstream sa_pout((root+"sa_p").c_str());
                 std::ofstream sa_sout((root+"sa_s").c_str());
                 //std::ofstream adjout((root+"adj").c_str());
                 std::ofstream tr_out((root+"tr").c_str());

//                 // Save demo for LRP/LPR
//                 save_demo(demo_ttt,root+"demo",0);

//                 // Save demo for MaxEnt/FIRL
//                 assert(env_ttt.term[659]==1);
//                 save_demo(demo_ttt,root+"demooth",1,9,659,0);

//                 // Saving virtual demo
//                 save_demo(v_demo_ttt,root+"v_demo",0);

//                 save_demo(v_demo_ttt,root+"v_demooth",1,9,inv_state[659],0);



                 // Saving the features for FIRL/Maxent

                 for(int v_s=0; v_s<v_n_states; v_s++){
                     VectorXmp& bas = env_ttt.get_v_basis(virtual_state[v_s]);

                     for(int i=0; i< env_ttt.get_v_num_training(); i++){

                         featout<<bas[i]<<"\t";
                     }
                     featout<<std::endl;
                 }

                 /// Features of the terminal states

                 VectorXmp tmp4=VectorXmp::Zero(env_ttt.get_v_num_training());

                 for(int i=0; i< env_ttt.get_v_num_training(); i++){

                     featout<<tmp4[i]<<"\t";
                 }
                 featout<<std::endl;





                 // Computing sa_p : transition probability

                 std::vector<std::vector<std::vector<double> > > sa_p(v_n_states+1,std::vector<std::vector<double> >(9,std::vector<double>(9,0))), sa_s(v_n_states+1,std::vector<std::vector<double> >(9,std::vector<double>(9,0))),tr(v_n_states+1,std::vector<std::vector<double> >(9,std::vector<double>(v_n_states+1,0)));

                 // Taking care of artificial terminal state
                 for(int a =0; a<9; a++){

                     sa_p[v_term_state][a][0] = 1;
                     sa_s[v_term_state][a][0] = v_term_state;
                     tr[v_term_state][a][v_term_state] = 1;


                 }
                 // We loop through the states
                 for(int v_s=0; v_s <v_n_states; v_s++){


                     int s = virtual_state[v_s];
                     if(env_ttt.term[s] == 1){

                         assert(env_ttt.transition[s].size()==0);

                         for(int a =0; a<9; a++){

                             sa_p[v_s][a][0] = 1;
                             //sa_s[v_s][a][0] = v_s;
                             sa_s[v_s][a][0] = v_term_state;
                             //tr[v_s][a][v_s] = 1;
                             tr[v_s][a][v_term_state] = 1;
                             int v_s_next = 0;
                             for(int k=1; k<9; k++){
                                 sa_p[v_s][a][k] = 0;
                                 sa_s[v_s][a][k] = v_s_next;
                                 v_s_next++;
                             }
                         }
                     }

                     else {

                         for(int all_a=0; all_a<9;all_a++){

                             int a = std::min(all_a, (int)env_ttt.transition[s].size()-1);
                             int v_a = env_ttt.transition[s][a];


                             if(env_ttt.term[v_a] == 1){

                                 sa_p[v_s][all_a][0] = 1;
                                 //sa_s[v_s][all_a][0] = v_s;
                                 //tr[v_s][all_a][v_s] = 1;
                                 sa_s[v_s][all_a][0] = v_term_state;
                                 tr[v_s][all_a][v_term_state] = 1;
                                 int v_s_next = 0;
                                 for(int k=1; k<9; k++){
                                     sa_p[v_s][all_a][k] = 0;
                                     sa_s[v_s][all_a][k] = v_s_next;
                                     v_s_next++;
                                 }

                             }
                             else{
                             vi tmp= env_ttt.transition[v_a];
                             for(int k=0; k<(int)tmp.size(); k++){

                                 sa_p[v_s][all_a][k] = 1.0/(double)tmp.size();
                                 sa_s[v_s][all_a][k] = inv_state[tmp[k]];
                                 tr[v_s][all_a][inv_state[tmp[k]]] = 1.0/(double)tmp.size();

                             }
                             int v_s_next =0;

                             for(int k = tmp.size(); k<9; k++){

                                 sa_p[v_s][all_a][k] = 0;
                                 sa_s[v_s][all_a][k]= v_s_next;
                                 v_s_next++;
                             }
                         }



                         }


                     }



                 }

                 //Checking that sa_p is correct

                 for(int v_s=0; v_s<v_n_states+1;v_s++){

                     for(int a=0; a<9; a++){

                         double total =0;

                         for(int k=0; k<9;k++){

                             total += std::abs(sa_p[v_s][a][k]);
                         }
                         std::cerr<<total<<" "<<v_s<<" ";
                         assert(std::abs(total-1) <= 1e-9);
                     }
                 }

                 // Saving sa_p in a file

                 for(int v_s=0; v_s <v_n_states+1;v_s++){

                     for(int a=0; a<9; a++){

                         for(int k=0; k<9;k++){

                             sa_pout<<sa_p[v_s][a][k]<<"\t";

                         }
                     }
                 }

                 // Saving sa_p for RPB n_states n_actions n_states in a file

                 for(int v_s=0; v_s <v_n_states+1;v_s++){

                     for(int a=0; a<9; a++){

                         for(int s_next=0; s_next<v_n_states+1;s_next++){

                             tr_out<<tr[v_s][a][s_next]<<"\t";

                         }
                     }
                 }

                 // Computing sa_s and saving in a file

                 for(int v_s=0; v_s <v_n_states+1;v_s++){

                     for(int a=0; a<9; a++){
                         for(int k=0; k<9;k++){

                             sa_sout<<sa_s[v_s][a][k]<<"\t";
                         }

                     }

                 }

//                 // Computing state adjacency

//                 for(int v_s=0; v_s<v_n_states; v_s++){

//                     for(int k = 0; k < 9; k++){

//                         bool isAdj = false;

//                         for(int a = 0; a<9; a++){
//                             if(sa_p[v_s][a][k] > 0){
//                                 isAdj = true;
//                                 break;
//                             }
//                         }

//                         if(isAdj){
//                             adjout<<"1"<<"\t";
//                         }
//                         else{
//                             adjout<<"0"<<"\t";
//                         }
//                     }
//                 }




                 //adjout.close();
                 sa_pout.close();
                 sa_sout.close();
                 //demout.close();
                 //othout.close();
                 featout.close();
                 tr_out.close();
                 break;
             }

             if(alg == ALG_OPT){
                 break;
             }


			std::cout << "Running MyIRL on the TTT environment" << std::endl;

			/*for (int s = 0; s < (int) alg_ttt.policy.size(); s++) {
			 if (alg_ttt.player[s] == 0 && env_ttt.term[s] == 0) {
			 int t2 = env_ttt.get_index_state(s, alg_ttt.policy[s]);
			 demo_ttt.Observe(s, t2, env_ttt.transition[s][t2]);
			 }
			 }*/

            if(alg == ALG_MANUAL){

                time_algo = 0;
                std::ifstream man_in((root+"manual_policy").c_str());
                greedy_pol = std::vector<int>(n_states,0);
                vi brut_pol;
                int t2;
                while(man_in>>t2){
                    brut_pol.push_back(t2);
                }
                std::cerr<<"n_states "<<n_states<<"v_n_states "<<v_n_states<<"brut "<<brut_pol.size()<<std::endl;
                //
                assert(brut_pol.size()==n_states || brut_pol.size()==v_n_states || brut_pol.size()==v_n_states+1);

                if(brut_pol.size() == n_states){

                    for(int s=0; s<n_states;s++){



                        greedy_pol[s] = brut_pol[s];


                        greedy_pol[s] = std::min(greedy_pol[s],(int)env_ttt.transition[s].size()-1 );
                        //greedy_pol[s] = std::max(greedy_pol[s],0);
                    }
                }

                else if(brut_pol.size() == v_n_states || brut_pol.size()==v_n_states+1){

                for(int v_s=0; v_s<v_n_states;v_s++){
                    int s = virtual_state[v_s];
                    //man_in>>greedy_pol[s];


                    greedy_pol[s] = brut_pol[v_s];

                    greedy_pol[s] = std::min(greedy_pol[s],(int)env_ttt.transition[s].size()-1 );
                    //greedy_pol[s] = std::max(greedy_pol[s],0);
                }
                }


            }
            else{

			IRL irl_ttt(env_ttt, demo_ttt,mode, &pi_demo_ttt, reg);



			// Testing approximator
			//std::cout<<"Testing approximator"<<std::endl;

			//NNApproximator *nn_ess =new NNApproximator(5);
			//FNNApproximator *fnn_ess= new FNNApproximator(5);
			//std::cout<<"True "<<nn_ess->get_num_training()<<std::endl;
			//std::cout<<"Mine "<<fnn_ess->get_num_training()<<std::endl;
			//assert(nn_ess->get_num_training() == fnn_ess->get_num_training());
			//VectorXd input =VectorXd::Random(5);
			//VectorXd wei = VectorXd::Random(nn_ess->get_num_training());
			//VectorXd wei = VectorXd::Ones(nn_ess->get_num_training());
			//std::cout<<"True "<<nn_ess->predict(input,wei)<<std::endl;
			//std::cout<<"Mine "<<fnn_ess->predict(input,wei)<<std::endl;

			//irl_ttt.computeMC(irl_policy, 720, 500000, IRL_CPRB);
			//irl_ttt.computeLRS(irl_policy, 720, 80000, IRL_CPRB,80000);
			//irl_ttt.computeBasicSPSA(irl_policy, 720, 50000, IRL_CPRB);
			//irl_ttt.computeGSS(irl_policy, 720, 80000, IRL_CPRB,80000);
			//computeOpt(irl_policy, env_ttt, demo_ttt,720,200,200000000,0.02,
			//	IRL_CPRB);
            if(optimisateur == OPT_LRP){

                time_algo = computeLRP(demo_ttt, env_ttt,irl_policy,alg);
                //irl_policy[17] = 40000;
                //irl_policy[23] = -15000;
                //irl_policy[12+3*2+12*12+9] = 40000;
                //irl_policy[12+3*2+12*12+9+1] = -15000;
			}
			else if(optimisateur != OPT_NLOPT)
			irl_ttt.optimize(irl_policy, max_time, n_iterations, mode, freq, optimisateur,useMean);

			else if(optimisateur == OPT_NLOPT){

				std::cerr<<"This option has been disabled"<<std::endl;
				exit(0);
						//computeOpt(irl_policy, env_ttt, demo_ttt, max_time, n_iterations, 20000, 0.002,mode);
					}

            greedy_pol = vi(n_states,-1);

        for (int s = 0; s < n_states; s++) {

            if (alg_ttt.player[s] == 0 && env_ttt.term[s] == 0) {

                greedy_pol[s]=env_ttt.policy_act(irl_policy, s);


            }


        }

}


			std::cout << " Displaying the found weight" << std::endl;

						///

						std::cout << "Computing states deviation" << std::endl;

						s_dev = 0;
						s_dev_tot = 0;

						std::cout<<"Policy Matrix"<<std::endl<<"["<<std::endl;

						for (int i = 0; i < (int) alg_ttt.policy.size(); i++) {

							if (alg_ttt.player[i] == 0 && env_ttt.term[i] == 0) {

                                //int j = env_ttt.policy_act(irl_policy, i);
                                int j = greedy_pol[i];
								s_dev_tot++;
                                if (env_ttt.get_state_index(i, j) != alg_ttt.policy[i] && (std::abs(alg_ttt.V(
                                        env_ttt.get_state_index(i, j))-alg_ttt.V(alg_ttt.policy[i]))>1e-9)  )
									s_dev++;
								std::cout<<j<<",";
                                //env_ttt.display(i);
                                //std::cerr<<std::endl;
                                //env_ttt.display(env_ttt.get_state_index(i, j));
                                //std::cerr<<std::endl;
                                //std::cerr<<std::endl;
								if(run == 0){

									if(i==(int) alg_ttt.policy.size()-1){
										policyout<<j<<std::endl;
									}
									else {
										policyout<<j<<"\t";
									}
								}
							}

						}
						if(run==0){
							policyout.close();
						}
						std::cout<<"]"<<std::endl;

						std::cout << "Deviates in " << s_dev << " states out of "
								<< s_dev_tot << std::endl;

						deviations[run] = s_dev;







			std::cout << " Evaluating the performance og myIRL" << std::endl;

			env_ttt.Reset();

            PolicyPerformance_app(env_ttt, greedy_pol, point, num, draw, loss,all_best_action,
                    false,false);

			std::cout << "Performance of my IRL" << std::endl;

			std::cout << point << " Points after " << num << " Plays"
					<< " with "
							"" << loss << " Games lost and " << draw << " Draw "
					<< std::endl;

			stat[run][1].reward += point;
			//stat[run][1].n_games += num;
			stat[run][1].num_draw += draw;
			stat[run][1].num_loss += loss;
			stat[run][1].name = "MyIRL";
			results[run] = point;

			stat[run][1].n_games += s_dev;
            win = num-draw-loss;

            save_stats(root,s_dev,num,point,draw,win,loss,time_algo,greedy_pol);


			break;
		}

		//std::cout << irl_policy << std::endl;

		//-------------------------------------------------------------------/

		// Starting the Blackjack environement  //

		/// Computing Optimal Policy



		while(toggleBL){
		AppBlackjack env_bl(approximateur,mode);

        //DiscreteMDP *mdp1 = env_bl.getMDP();
        //int n_states = mdp1->getNStates();
        int n_states = env_bl.get_n_states();
        // To keep the greedy policy
        std::vector<int> greedy_pol;
		//int n_actions = mdp1->getNActions();
		/*std::vector<std::vector<double> > prob(n_states,
				std::vector<double>(n_actions, 0));
		std::vector<std::vector<double> > num12(n_states,
				std::vector<double>(n_actions, 0));

		/// Estimating the True reward probability by running the games multiple time

		action_ok = false;
		for (int i = 0; i < 100000; i++) {
			if (!action_ok) {
				env_bl.reset();
				action_ok = true;

			}

			int action = rand() % 2;

			int state = env_bl.get_state();
			action_ok = env_bl.act(action);
			int reward = env_bl.get_reward();
			//if(action==0 && reward!=0)
			//std::cout<<"ac "<<action<<"\t";
			prob[state][action] += reward;
			num12[state][action] += 1;

		}

		for (int i = 0; i < n_states; i++) {
			for (int j = 0; j < n_actions; j++)

			{
				if (num12[i][j] != 0) {
					mdp1->setFixedReward(i, j, prob[i][j] / num12[i][j]);
					std::cout << prob[i][j] / num12[i][j] << "\t";
				}

				if (i == n_states - 1)
					mdp1->setFixedReward(i, j, 0);

			}
		}

		/// Computing the optimal policy

		ValueIteration VI(mdp1, 1);
		VI.ComputeStateValues(0.1 * 0.000000000001, -1);
		FixedDiscretePolicy alg_bl(n_states, n_actions, VI.Q);
		alg_bl.MakeGreedyPolicy();*/


		/// PreComputing the optimal Policy according to paper


		std::vector<int> opt_sol(201, 0);

		opt_sol[3] = 1;
		opt_sol[4] = 1;
		opt_sol[5] = 1;

		for (int i = 13; i <= 16; i++)
			for (int j = 1; j < 6; j++)
				opt_sol[(i - 12) * 10 + j] = 1;

		for (int i = 17; i <= 21; i++)
			for (int j = 0; j < 10; j++)
				opt_sol[(i - 12) * 10 + j] = 1;

		for (int i = 19; i <= 21; i++)
			for (int j = 0; j < 10; j++)
				opt_sol[(i - 12) * 10 + 100 + j] = 1;
		for (int i = 1; i < 8; i++)
			opt_sol[(18 - 12) * 10 + 100 + i] = 1;

        Demonstration demo_bl, pi_demo_bl;

        if(demo_file ==""){

		std::cout << "Observing the Optimal Policy of Blackjack" << std::endl;


		action_ok = false; //
		episode = -1;
		start1 = false;

		for (int step = 0; step < n_step; step++) {

			if (!action_ok) {

				episode++;

                if (episode >= n_episodes) {

					break;
				}

				else {

					env_bl.reset();
					action_ok = true;

                    if(episode >= limit_ep){
                        demo_bl.NewEpisode();

                    }
				}
			}

			int state = env_bl.get_state();
			//real reward = env_ttt.getReward();
			start1 = true;

			int action = 0;
			//alg_bl.Observe(0, state);
			//action = alg_bl.SelectAction();
			action = opt_sol[state];

			//int action_index = env_ttt.get_index_state(state, action);

			action_ok = env_bl.act(action);
			int next_state = env_bl.get_state();
			//reward = env_ttt.getReward();

            if (episode >= limit_ep) {


				//if(action_ok)
				demo_bl.Observe(state, action, next_state);
				//else
					//demo_bl.Observe(state, action, -1);

			}


		}

        assert(demo_bl.size()==n_episodes);



		/// Observing demonstrations for the computing of the approximate optimal policy


		if(randomPIdemo){

			action_ok = false; //
					episode = -1;
					start1 = false;

					for (int step = 0; step < n_step; step++) {

						if (!action_ok) {


							episode++;

                            if (episode >= num_PIdemo) {

								break;
							}

							else {

								env_bl.reset();
								action_ok = true;

                                if(episode >= limit_ep){
                                    pi_demo_bl.NewEpisode();
                                }
							}
						}

						int state = env_bl.get_state();
						//real reward = env_ttt.getReward();
						start1 = true;

						int action = 0;
						//alg_bl.Observe(0, state);
						//action = alg_bl.SelectAction();
						//action = opt_sol[state];
						if(env_bl.get_n_actions(state) > 0)
						action = rand()%(env_bl.get_n_actions(state));

						//int action_index = env_ttt.get_index_state(state, action);

						action_ok = env_bl.act(action);
						int next_state = env_bl.get_state();
						//reward = env_ttt.getReward();

                        if (episode >= limit_ep) {


							//if(action_ok)
							pi_demo_bl.Observe(state, action, next_state);
							//else
								//demo_bl.Observe(state, action, -1);

						}


					}


		}
		else {

			pi_demo_bl = demo_bl;
		}

        }

        // a demonstration file has been given
        else{

            std::cerr<<"We will load a pre computed demonstration, randompidemo does not work with this mode currently"<<std::endl;

            demo_bl = load_demo(demo_file);

            // Hack we should remove randompidemo later
            pi_demo_bl = demo_bl;

            //demo_in.close();


        }

		std::cout << "Performance of the optimal Blackjack Policy" << std::endl;

		vi policy_bl(n_states);

		for (int i = 0; i < n_states; i++) {
			//alg_bl.Observe(0, i);
			//int action = alg_bl.SelectAction();
			int action = opt_sol[i];
			policy_bl[i] = action;
		}

        //PolicyPerformance_app(env_bl, policy_bl, point, num, draw, loss,false,true);

        // Computing performance of optimal policy only when required
        if(alg == ALG_OPT || (demo_file == "" && alg != ALG_MANUAL)){
        PolicyPerformance(env_bl, opt_sol, point, num, draw, loss);

		std::cout << point << " Points after " << num << " Plays" << " with "
				"" << loss << " Games lost and " << draw << " Draw "
				<< std::endl;

		stat[run][0].reward += point;
		//stat[run][0].n_games += num;
		stat[run][0].num_draw += draw;
		stat[run][0].num_loss += loss;
		stat[run][0].name = "OptimalBL";
		stat[run][0].n_games += 0;
        time_algo = 0;
        win = num-draw-loss;
        }

        // If we are only testing the optimal algo
        // we don't need to go further
        if(alg == ALG_OPT){

            s_dev = 0;
            greedy_pol = opt_sol;

            // Saving statistics for optimal policy
            save_stats(root,s_dev,num,point,draw,win,loss,time_algo,greedy_pol);


            //std::ofstream demout((root+"demo").c_str());
            //std::ofstream othout((root+"demooth").c_str());
            std::ofstream featout((root+"feat").c_str());
            std::ofstream sa_pout((root+"sa_p").c_str());
            std::ofstream sa_sout((root+"sa_s").c_str());
            std::ofstream adjout((root+"adj").c_str());

            // Save demo for LRP/LPR
            save_demo(demo_bl,root+"demo",0);

            // Save demo for MaxEnt/FIRL
            save_demo(demo_bl,root+"demooth",1,7,200,0);

            // Saving the features for FIRL/Maxent

            for(int s=0; s<n_states; s++){
                VectorXmp& bas = env_bl.get_v_basis(s);

                for(int i=0; i< env_bl.get_v_num_training(); i++){

                    featout<<bas[i]<<"\t";
                }
                featout<<std::endl;
            }

            // Computing sa_p : transition probability

            std::vector<std::vector<std::vector<double> > > sa_p(n_states,std::vector<std::vector<double> >(2,std::vector<double>(n_states,0)));

            // Probability of terminal_state
            sa_p[200][0][200] = 1;
            sa_p[200][1][200] = 1;

            // We loop through the states without the terminal_state
            for(int s=0; s <n_states-1; s++){

                // Simulating action 0
                for(int card=1; card<11; card++){

                    double pr = 0.0;

                    if(card==10){
                        pr = 4.0/13.0;
                    }
                    else{
                        pr = 1.0/13.0;
                    }
                    int pc,dc,isU;
                    env_bl.getTriplet(pc,dc,isU,s);
                    int tmp_pc = pc+card;
                    if(tmp_pc > 21 && isU == 1){
                        tmp_pc -=10;
                        isU = 0;
                    }

                    int s_next = env_bl.getState(tmp_pc, dc, isU);
                    sa_p[s][0][s_next] += pr;


                }

                // Simulating action 1

                sa_p[s][1][200] = 1;



            }

            //Checking that sa_p is correct

            for(int s=0; s<n_states;s++){

                for(int a=0; a<2; a++){

                    double total =0;

                    for(int s_next=0; s_next<n_states;s_next++){

                        total += std::abs(sa_p[s][a][s_next]);
                    }
                    std::cerr<<total<<" "<<s<<" ";
                    assert(std::abs(total-1) <= 1e-9);
                }
            }

            // Saving sa_p in a file

            for(int s=0; s <n_states;s++){

                for(int a=0; a<2; a++){

                    for(int s_next=0; s_next<n_states;s_next++){

                        sa_pout<<sa_p[s][a][s_next]<<"\t";

                    }
                }
            }

            // Computing sa_s and saving in a file

            for(int s=0; s <n_states;s++){

                for(int a=0; a<2; a++){
                    for(int s_next=0; s_next<n_states;s_next++){

                        sa_sout<<s_next<<"\t";
                    }

                }

            }

            // Computing state adjacency

            for(int s=0; s<n_states; s++){

                for(int s_next = 0; s_next < n_states; s_next++){

                    if(sa_p[s][0][s_next] > 0 || sa_p[s][1][s_next] > 0){
                        adjout<<"1"<<"\t";
                    }
                    else{
                        adjout<<"0"<<"\t";
                    }
                }
            }




            adjout.close();
            sa_pout.close();
            sa_sout.close();
            //demout.close();
            //othout.close();
            featout.close();
            break;
        }

		std::cout << "Computing MyIRL on the Blackjack" << std::endl;


        if(alg == ALG_MANUAL){

            time_algo = 0;
            std::ifstream man_in((root+"manual_policy").c_str());
            greedy_pol = std::vector<int>(n_states,0);
            for(int s=0; s<n_states;s++){
                man_in>>greedy_pol[s];
            }


        }
        else {
		IRL irl_bl(env_bl, demo_bl,mode, &pi_demo_bl, reg);
		VectorXd irl_policy_bl;

		//irl_bl.computeMC(irl_policy_bl, 720, 80000, IRL_CRPB);
		//irl_bl.computeLRS(irl_policy_bl, 720, 60000, IRL_CRPB, 2920000);
		//irl_bl.computeGSS(irl_policy_bl, 720, 10000, IRL_CRPB, 20000);
		//computeOpt(irl_policy_bl, env_bl, demo_bl);
		//irl_bl.computeBasicSPSA(irl_policy_bl, 720, 20000, IRL_CRPB);
		if(optimisateur == OPT_LRP){

               time_algo = computeLRP(demo_bl, env_bl,irl_policy_bl,alg);
			}

		else if(optimisateur != OPT_NLOPT )
		irl_bl.optimize(irl_policy_bl, max_time, n_iterations, mode, freq, optimisateur,useMean);

		else if(optimisateur == OPT_NLOPT){
			std::cerr<<"This option has been disabled"<<std::endl;
			exit(0);
			//computeOpt(irl_policy_bl, env_bl, demo_bl, max_time, n_iterations, 20000, 0.002,mode);
		}

        std::cout << " Displaying the found weight" << std::endl;

        std::cout << irl_policy_bl << std::endl;

        // Converting the Q function to a greedy policy
        greedy_pol = env_bl.greedy_pol(irl_policy_bl);

        }

        std::cout << " Evaluating the performance of myIRL" << std::endl;

		env_bl.reset();

        //PolicyPerformance_app(env_bl, irl_policy_bl, point, num, draw, loss, true);
        PolicyPerformance(env_bl, greedy_pol, point, num, draw, loss);


		std::cout << "Performance of my IRL" << std::endl;

		std::cout << point << " Points after " << num << " Plays" << " with "
				"" << loss << " Games lost and " << draw << " Draw "
				<< std::endl;

		stat[run][1].reward += point;
		//stat[run][1].n_games += num;
		stat[run][1].num_draw += draw;
		stat[run][1].num_loss += loss;
		stat[run][1].name = "MyIRLBL";
        win = num-draw-loss;

		results[run]=point;



		// Number of states deviation

		std::cout << "Number of states deviation" << std::endl;
		s_dev = 0;

		if(run>=0){
			std::cout<<"Policy Matrix"<<std::endl;

			std::cout<<"[";
		}



		for (int i = 0; i < n_states; i++) {

			if (i != env_bl.terminal_state) {

                //int j = env_bl.policy_act(irl_policy_bl, i);
                int j = greedy_pol[i];

				if (env_bl.get_state_index(i, j) != policy_bl[i])
					s_dev++;

				if(run>=0){

					std::cout<<env_bl.get_state_index(i, j)<<",";
				}

				if(run == 0){

					policyout<<env_bl.get_state_index(i, j)<<"\t";
				}

			}
			else {

				if(run>=0){

				std::cout<<env_bl.get_state_index(i, 0);
				}
				if(run==0){
					policyout<<env_bl.get_state_index(i, 0)<<std::endl;
				}


			}

		}

		if(run == 0){
			policyout.close();
		}
		if(run>=0)

		std::cout<<"]"<<std::endl;

		std::cout << "Deviate in " << s_dev << " states out of " << n_states - 1
				<< std::endl;
		deviations[run] = s_dev;
		stat[run][1].n_games += s_dev;

        save_stats(root,s_dev,num,point,draw,win,loss,time_algo,greedy_pol);


		//FixedSoftmaxPolicy softmax_policy(VI.Q, 5000);
		break;
		}

		}
	}


	for(int run = 0; run <n_run; run++){

	std::cout << "Algorithm\t" << " Number of points\t" << " Number of Loss\t"
			<< " Number of Draw\t" << " Number of Deviations\t" << std::endl;
	for (int i = 0; i < stat_num; i++) {

		stat[run][i].reward /= freqForOneRun;
		stat[run][i].n_games /= freqForOneRun;
		stat[run][i].num_draw /= freqForOneRun;
		stat[run][i].num_loss /= freqForOneRun;
		std::cout << stat[run][i].name << "\t\t" << stat[run][i].reward << "\t\t\t"
				<< stat[run][i].num_loss << "\t\t" << stat[run][i].num_draw << "\t\t\t"
				<< stat[run][i].n_games << std::endl;

	}
	}

	std::ofstream evolout((root+"NumberofPoints.txt").c_str());
	std::ofstream devout((root+"NumberofDeviations.txt").c_str());

	std::cout<<"[";

	for(int i=0; i< (int)results.size();i++){

		std::cout<<stat[i][1].reward<<",";
		//std::cout<<deviations[i]<<",";

		if(i==(int)results.size() - 1){
			//evolout<<results[i]<<std::endl;
			evolout<<stat[i][1].reward<<std::endl;
			//devout<<deviations[i]<<std::endl;
			devout<<stat[i][1].n_games<<std::endl;
		}
		else {
			//evolout<<results[i]<<"\t";
			evolout<<stat[i][1].reward<<"\t";
			//devout<<deviations[i]<<"\t";
			devout<<stat[i][1].n_games<<"\t";
		}
	}
	evolout.close();
	devout.close();
	std::cout<<"]"<<std::endl;



	std::cout<<"[";

		for(int i=0; i< (int)results.size();i++){

			//std::cout<<results[i]<<",";
			std::cout<<stat[i][1].n_games<<",";

		}
		std::cout<<"]"<<std::endl;
	std::cout << " After " << n_run << " Experiments " << std::endl;

	std::cout << "# Done" << std::endl;

	return 0;
}
