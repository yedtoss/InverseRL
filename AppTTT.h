/*
 * AppTTT.h
 *
 *  Created on: 8 Sep 2012
 *      Author: yedtoss
 */

#ifndef APPTTT_H_
#define APPTTT_H_

#include "AppEnvironment.h"


/**
Tic-Tac-Toe environment
0 means empty case
1 first player
2 second player

*/

#include "Matrix.h"
#include "Vector.h"
//#include "real.h"
#include <vector>
#include "MG.h"
#include <set>
#include "Utils.h"
//using namespace std;
typedef std::vector<int> vi;
typedef std::vector<std::vector<int> > vii;


class AppTTT: public AppEnvironment {
public:

	vi player, rew, term;
	std::vector<std::set<int> > succ; /// To keep the sucessors in set for fast searching
	//int n_states;
	//int state;
	//double reward;
	Matrix F4; //basis;
	vii sol;
	MG * mg; // To keep the current Markov game
	std::vector<VectorXd> v_basis;
	std::vector<VectorXmp> v_basis_bis;
	vii transition;

    AppTTT(App type_, bool use_kernel_=false, Demonstration *D_=NULL, bool MultipleNN_ = false ,IRL_MODE mode = IRL_CPRB, int num_hidden = 1, vi layerlist = vi(1,5)) {
		n_states = 19683;
		n_features = 9;
		n_features = 13;
		type =type_;
        use_kernel = use_kernel_;
		//n_features = 9;
		vi t(9, 0);
		//app = new NNApproximator(n_features);
		//app = new LinearApproximator(n_features);
		//app = new FNNApproximator(n_features);
		//vi netw(3,n_features);
		//netw[2] = 1;

		if(type == App_Linear){
            //n_features = 4;
			//n_features = 13;
			//n_features = 9;
            //n_features = 8+9+0+(9+8)*4;
            n_features = 10+10*10+9;
           // n_features = 8+9+(8+9)*(8+9)+1;

			app = new LinearApproximator(n_features);
			qapp = new LinearApproximator(n_features);
			MultipleNN = false;
		}
		else if(type == App_NN){
			//n_features = 13;
			n_features = 4;
			//n_features = 9;
			vi netw(2+num_hidden,n_features);

			for(int i = 0; i < num_hidden; i++){

				netw[1+i] = layerlist[i];
			}
			//netw[1] = 2*n_features;
			//netw[2] = 9;
			netw[netw.size()-1] = 1;
			app = new FNNApproximator(netw, ACT_TANH);
			qapp = new FNNApproximator(netw, ACT_TANH);
			//app = new NNApproximator(netw);
			//app = new LinearApproximator(n_features);
			//qapp = new NNApproximator(netw);
			MultipleNN = false;

		}


		vapp = app;
		rapp = app;
		//vii sol;
		//sol.push_back(t);
		//backtrack(t,sol,0,0);

		///  b2
		std::set<vi> sol2;
		backtrack2(t, sol2, 0); /// Creating all possible unique valid or not states

		/// This routines eliminate all non valid states of TTT
		for (std::set<vi>::iterator it = sol2.begin(); it != sol2.end(); it++) {
			int c2 = 0, c = 0, c0 = 0;

			/// Computing number of 0, 1 and 2 in the grid
			for (int i = 0; i < 9; i++) {
				if ((*it)[i] == 2)
					c2++;
				else if ((*it)[i] == 1)
					c++;
				else
					c0++;

			}

			/// Eliminating non valid grids
			/// For valid grid Computing their reward and if they are terminal or not
			if (c == c2 || c == c2 + 1) {
				vi tmp = *it;
				int c3 = isT2(tmp);
				if (c3 < 3) {
					sol.push_back(*it);
					player.push_back((int) c != c2);
					if (c3 != 0) {
						term.push_back(1);
					} else {
						if (c0 == 0)
							term.push_back(1);
						else
							term.push_back(0);
					}
					if (c3 < 2)
						rew.push_back(c3);
					else
						rew.push_back(-1);
				}
			}
		}
		std::cout << " Backtrack end " << std::endl;

		int nsol = sol.size();
		//std::cout << " nsol " << nsol << std::endl;
		n_states = nsol;
		succ.resize(nsol);
		/*player.resize(nsol,0);
		 rew.resize(nsol,0);
		 term.resize(nsol,0);
		 //display();




		 for(int i=0;i<nsol;i++)
		 {
		 int c=0;
		 for(int k=0;k<9;k++)
		 {
		 if(sol[i][k]!=0)
		 c++;
		 }
		 if(c%2==0)
		 player[i]=0;
		 else
		 player[i]=1;

		 int c2=isT(sol[i]);
		 if(c2!=0)
		 {
		 term[i]=1;
		 rew[i]=c2;
		 if(c2==2)
		 rew[i]=-1;
		 }
		 else
		 {
		 if(c==9)
		 term[i]=1;
		 else
		 term[i]=0;
		 rew[i]=0;
		 }

		 }*/

		//std::cout << " Succ begin " << std::endl;

		/// This routine computes the successor for each grid
		vi used(nsol, 0);
		for (int i = 0; i < nsol; i++) {

			if (term[i] == 1)
				continue;

			int p = 0, p2 = 0;
			int n1 = 0, n2 = 0;
			for (int k = 0; k < 9; k++) {
				if (sol[i][k] == 0)
					p++;
				if (sol[i][k] == 1)
					n1++;
				if (sol[i][k] == 2)
					n2++;
			}

			for (int j = 0; j < nsol; j++) {
				if (p2 == p)
					break;
				if (j != i && used[j] == 0 && player[j] != player[i]) {
					int diff = 0;
					int nd = 0, nb = 0;

					for (int k = 0; k < 9; k++) {
						if (sol[i][k] != sol[j][k]) {
							diff++;
							nd = sol[i][k];
							nb = sol[j][k];

						}

					}
					if (diff == 1) {
						if (nd == 0) {
							if (nb == 1) {
								if (n1 == n2) {
									succ[i].insert(j);
									p2++;
								}

							}
							if (nb == 2) {
								if (n1 == n2 + 1) {
									succ[i].insert(j);
									p2++;
								}

							}

						}
						//used[j]=2;

					}

				}
			}
		}
		//std::cout << " Succ end " << std::endl;

		// affiche succ 78
		//for(std::set<int>::iterator it3=succ[78].begin();it3!=succ[78].end();it3++)
		//display(*it3);
		//std::cout << "rew 92-- " << rew[91] << std::endl;
		//display();

		/// Computing the successors in a vector

		transition.resize(nsol);
		for (int i = 0; i < nsol; i++) {

			transition[i] = std::vector<int>(succ[i].begin(), succ[i].end());
		}

		//std::cout << "T size = " << transition.size() << std::endl;

		/// Computing the basis

		//getBasis();
        //getBasis2();
        getBasis3();

		if(n_features<9)
			compute_basis_4(n_features);
		if(n_features>=9)
			compute_basis_9(n_features);

	}


	int isT(vi &t) {
		if (t[0] == t[1] && t[1] == t[2] && t[0] != 0)
			return t[0];
		if (t[3] == t[4] && t[4] == t[5] && t[3] != 0)
			return t[3];
		if (t[6] == t[7] && t[7] == t[8] && t[6] != 0)
			return t[6];
		if (t[0] == t[3] && t[3] == t[6] && t[0] != 0)
			return t[0];
		if (t[1] == t[4] && t[4] == t[7] && t[1] != 0)
			return t[1];
		if (t[2] == t[5] && t[5] == t[8] && t[2] != 0)
			return t[2];
		if (t[0] == t[4] && t[4] == t[8] && t[0] != 0)
			return t[0];
		if (t[2] == t[4] && t[4] == t[6] && t[2] != 0)
			return t[2];
		return 0;

	}

	int isT2(vi &t) {
		int res = 0;
		int last = 0;
		if (t[0] == t[1] && t[1] == t[2] && t[0] != 0) {

			if (t[0] != last && last != 0)
				return 3;
			last = t[0];
			res++;
		}

		if (t[3] == t[4] && t[4] == t[5] && t[3] != 0) {

			if (t[3] != last && last != 0)
				return 3;
			last = t[3];
			res++;
		}
		if (t[6] == t[7] && t[7] == t[8] && t[6] != 0) {

			if (t[6] != last && last != 0)
				return 3;
			last = t[6];
			res++;
		}
		if (t[0] == t[3] && t[3] == t[6] && t[0] != 0) {

			if (t[0] != last && last != 0)
				return 3;
			last = t[0];
			res++;
		}
		if (t[1] == t[4] && t[4] == t[7] && t[1] != 0) {

			if (t[4] != last && last != 0)
				return 3;
			last = t[4];
			res++;
		}
		if (t[2] == t[5] && t[5] == t[8] && t[2] != 0) {

			if (t[2] != last && last != 0)
				return 3;
			last = t[2];
			res++;
		}
		if (t[0] == t[4] && t[4] == t[8] && t[0] != 0) {

			if (t[0] != last && last != 0)
				return 3;
			last = t[0];
			res++;
		}
		if (t[2] == t[4] && t[4] == t[6] && t[2] != 0) {

			if (t[2] != last && last != 0)
				return 3;
			last = t[2];
			res++;
		}

		return last;

	}

	/// This function lists all possible ( valid or not states of the games )

	void backtrack2(vi& t, std::set<vi> &sol, int k) {

		sol.insert(t);
		if (k != 9) {
			for (int i = 0; i < 3; i++) {
				t[k] = i;
				backtrack2(t, sol, k + 1);
			}

		}
	}

	/// This function all possible valid state only

	void backtrack(vi& t, vii &sol, int k, int pos, int pos2 = 0) {
		int c2 = isT2(t);

		int c = 0;
		if (k != 9 && c == 0) {
			if (c2 <= 2)

				sol.push_back(t);
		}
		if (k == 9 || c != 0) {
			if (c2 <= 2)
				sol.push_back(t);

		}

		else {

			int actu = pos;
			if (k % 2 == 0) {
				actu = pos;
			}

			else {
				actu = pos2;
			}

			for (int i = actu; i < 9; i++) {
				if (t[i] != 0)
					continue;
				if (k % 2 == 0) {
					t[i] = 1;
					backtrack(t, sol, k + 1, i + 1, pos2);
				} else {
					t[i] = 2;
					backtrack(t, sol, k + 1, pos, i + 1);
				}

				t[i] = 0;
			}

		}
	}

	/// Returning a pointer to the markov game

	MG *getMDP() {
		mg = new MG(rew, succ, player);
		return mg;

	}

	/// Semi- Random Player:  Player randomly unless he will loose or win; in which case he
	/// makes the right move

	int AIplay(bool semi=false) {


		if (succ[state].size() != 0) {

			/// attack: Note that with -1  the ai wins



			std::set<int>::iterator it = succ[state].begin(), it2;

			if(semi){
			for (it = succ[state].begin(); it != succ[state].end(); it++) {
				if (rew[*it] == -1) {
					state = *it;
					return *it;
				}
			}
			}

			/// Defence
			/*for (it = succ[state].begin(); it != succ[state].end(); it++) {
				for (it2 = succ[*it].begin(); it2 != succ[*it].end(); it2++) {
					if (rew[*it2] == 1) {
						break;
					}

					/// You should take the iterator just before the end
					if (it2 == succ[*it].end()) {
						state = *it;
						return *it;
					}
				}
			}*/

			/// random play

			int pos = rand() % (succ[state].size());
			it = succ[state].begin();
			int p = 0;
			while (p < pos) {
				it++;
				p++;
			}
			state = *it;
			return *it;

		}
		return 0;

	}

	/*bool act(int s) {
		return act(s, true, 0);
	}*/

	bool act(int s, bool autom = true, int s2 = -1) {
        //std::set<int>::iterator it = succ[state].find(s);
        int action2 = s;
        s = transition[state][s];
		//std::cout<<"s = "<<s<<" st= "<<state<<endl;
        //if (it != succ[state].end()) {
        if (action2 < (int)transition[state].size()) {

			reward = rew[s];
			state = s;
			if (term[s] == 1)
				return false;
			else {
				if (autom) {
					/*display(state);
					 cout<<"Play"<<endl;
					 for(set<int>::iterator it2=succ[state].begin();it2!=succ[state].end();it2++)
					 display(*it2);*/

					state = AIplay(s2>0);
					//cin>>state;
				} else
                {
                    //state = s2;

                    if(transition[s].size()>0)
                        state = transition[s][s2];
                    else{
                        std::cerr<<"should never come here"<<std::endl;
                        exit(-1);
                    }
                }


				if (term[state] == 1) {
					reward = rew[state];
					return false;
				}
				reward = rew[state];

				return true;
			}

		} else {
			for (std::set<int>::iterator it = succ[s].begin(); it
					!= succ[s].end(); it++) {
				std::cout << "s   " << *it << "  ";
			}
			reward = -5;
			std::cout << " Error ";
			exit(0);
			return false;

		}
	}

	/// To remove

	int getState() {
		return state;
	}
	/// To remove
	double getReward() {
		return reward;
	}
	/// to remove

	void Reset() {
		//std::cout<<"Bien\t";
		state = 0;
		reward = 0;
	}
	void reset(){

		state = 0;
		reward = 0;
	}

	std::string Name() {
		return "Tic-Tac-Toe";
	}

	/// Returning the vector of reward
	vi getRew() {
		return rew;
	}


	Matrix getBasis2(){
		F4.Resize(n_states, 4);


		// stat[i] contains the statistics for characteristic i

		/// stat[i][0] is the min value
		/// stat[i][1] is the max value
		/// stat[i][2] is the mean value


		std::vector<std::vector<double> > stat(4,std::vector<double>(3,0));



		for (int s = 0; s < n_states; s++) {

			// Status is a 2 dimensional matrix
			// Each row corresponds to a row, column or diagonal in the board
			// Precisely the first 3 row corresponds to the 3 rows of the board
			// The next 3 row corresponds to the 3 columns of the board starting from
			// the left, The next 2 rows contains the diagonal

			// Each columns of status indicates a number for the two players and for blank
			// status(i,j) indicates the number of points player j has placed on the row i
			std::vector<std::vector<int> > status(8,std::vector<int>(3,0));


			for(int k =0; k< 9; k++){

				// Updating a row

				status[k/3][sol[s][k]]++;

				// Updating the column corresponding to k

				status[k%3+3][sol[s][k]]++;


				// Updating the diagonal corresponding to k

				// First diagonal
				if(k%4 == 0){

					status[6][sol[s][k]]++;
				}

				// Second diagonal
				if(k%4==2 || k==4){

					status[7][sol[s][k]]++;
				}
			}


			///  Now we will count the number of singlets, doublets for each players
			/// F4(s,0) is the number of singlets for player 1
			/// F4(s,1) is the number of doublets for player 1

			/// F4(s,2) and F4(s,3) similarly for the second player

			for(int i=0; i< 4; i++){

				F4(s,i)=0;
			}


			for(int l=0; l<8; l++){

				// Singlets for player one

				if(status[l][0]==2 && status[l][1]==1){
					F4(s,0)++;
				}

				// Doublets for player one

				if(status[l][0]==1 && status[l][1] == 2){
					F4(s,1)++;
				}

				// Singlet for player 2

				if(status[l][0]==2 && status[l][2]==1){
									F4(s,2)++;
								}

				// Doublet for player 2

								if(status[l][0]==1 && status[l][2] == 2){
									F4(s,3)++;
								}


			}

			for(int i=0; i< 4; i++){

				stat[i][2]+=F4(s,i);
				if(s==0 || F4(s,i)<stat[i][0]){
					stat[i][0] = F4(s,i);
				}

				if(s==0 || F4(s,i)>stat[i][1]){

					stat[i][1] = F4(s,i);
				}
			}
		}


		/// Normalizing each feature

		/*for(int s=0; s<n_states;s++){

			for(int i=0; i<4;i++){

				F4(s,i)=(F4(s,i)-stat[i][2]/(double)n_states)/(stat[i][1]-stat[i][0]);
			}
		}*/

		return F4;

	}

	/// TODO change the type of return

	Matrix getBasis() {
		F4.Resize(n_states, 4);

		for (int s = 0; s < n_states; s++) {
			int lr = 0;
			// Counting elements in the 3 available rows
			std::vector<int> nr(3, 0);

			// nc for the columns, nd for the diagonals
			std::vector<std::vector<int> > nc(3, std::vector<int>(3, 0)), nd(2,
					std::vector<int>(3, 0));

			// Initializing the features to zero

			for (int i = 0; i < 4; i++)
				F4(s, i) = 0;

			for (int k = 0; k < 9; k++) {

				// Taking the rows
				if (k / 3 != lr) {


					if (nr[1] == 0 && nr[2] != 0 && nr[2] < 3) {
						F4(s, 1 + nr[2]) += 1;
					}
					if (nr[1] != 0 && nr[2] == 0 && nr[1] < 3) {
						F4(s, -1 + nr[1]) += 1;
					}
					nr[1] = 0;
					nr[2] = 0;
					lr = k / 3;
				}

				// sol[s][k] contains number identifying the player occupying the case k

				nr[sol[s][k]]++;
				nc[k % 3][sol[s][k]]++;
				if (k != 4) {
					if (k % 4 == 0 || k % 4 == 2)
						nd[(k % 4) / 2][sol[s][k]]++;
				} else {
					nd[0][sol[s][k]]++;
					nd[1][sol[s][k]]++;
				}

			}

			for (int i = 0; i < 3; i++) {
				if (nc[i][1] == 0 && nc[i][2] != 0 && nc[i][2] < 3) {
					F4(s, 1 + nc[i][2]) += 1;
				}
				if (nc[i][1] != 0 && nc[i][2] == 0 && nc[i][1] < 3) {
					F4(s, -1 + nc[i][1]) += 1;
				}

			}

			for (int i = 0; i < 2; i++) {
				if (nd[i][1] == 0 && nd[i][2] != 0 && nd[i][2] < 3) {
					F4(s, 1 + nd[i][2]) += 1;
				}
				if (nd[i][1] != 0 && nd[i][2] == 0 && nd[i][1] < 3) {
					F4(s, -1 + nd[i][1]) += 1;
				}

			}
		}

		return F4;
	}

    TTT_FEAT feat(std::vector<TTT_CASE>& board){

        // Counting the number of case for each player

        std::vector<int> count_p(3,0);

        for(int i=0; i< board.size();i++){
            count_p[board[i].player]++;

        }

        TTT_FEAT res;
        res.type = TTT_OTHER;
        res.player = TTT_PE;

        if(count_p[TTT_PE]==2){

            if(count_p[TTT_PO] == 1)
                res.player = TTT_PO;
            else if(count_p[TTT_PX] == 1)
                res.player = TTT_PX;
            res.type = TTT_SINGLET;
        }

       else if(count_p[TTT_PE] == 1){
            if(count_p[TTT_PO] == 2){
                res.player = TTT_PO;
                res.type = TTT_DOUBLET;
            }
            else if(count_p[TTT_PX] == 2){
                res.player = TTT_PX;
                res.type = TTT_DOUBLET;
            }



        }

        else if(count_p[TTT_PE] == 0){

            if(count_p[TTT_PO] == 3){
                res.player = TTT_PO;
                res.type = TTT_TRIPLET;
            }

            else if(count_p[TTT_PX] == 3){
                res.player = TTT_PX;
                res.type = TTT_TRIPLET;

            }
        }





        return res;
    }

    void update(std::vector<TTT_CASE>& board, std::vector<std::vector<int> >& singlet_count, std::vector<std::vector<int> >& features, std::vector<std::vector<int> >& diversity, std::vector<std::vector<int> >& opportunity,int s){

        TTT_FEAT res = feat(board);

        if(res.type == TTT_SINGLET){

            for(int col =0; col<board.size();col++){
               // if(board[col].player == TTT_PX || board[col].player == TTT_PO){
                if(board[col].player == TTT_PE){
                    singlet_count[board[col].case_id][res.player]++;

                }

                // I disable it to check if everything is right
                // I should enable it
                //if(board[col].player ==  res.player){
                if(board[col].player == TTT_PX || board[col].player == TTT_PO){
                    assert(res.player == board[col].player);

                    diversity[board[col].case_id][res.player]++;
                }

            }
        }

        else if(res.type == TTT_DOUBLET){

            for(int col =0; col<board.size();col++){

            if(board[col].player == TTT_PX || board[col].player == TTT_PO){

                opportunity[board[col].case_id][res.player]++;

            }
            }

        }

        if(res.type == TTT_DOUBLET || res.type == TTT_SINGLET || res.type == TTT_TRIPLET){
            features[res.type][res.player]++;
        }


//        if(res.type == TTT_TRIPLET){
//            std::cerr<<s<<"\t";
//            display(s);
//            assert(term[s] == 1);
//            if(res.player == TTT_PO){
//                std::cerr<<s<<"\t";
//                display(s);
//                assert(rew[s]==-1);
//            }
//            if(res.player == TTT_PX){
//                std::cerr<<s<<"\t";
//                display(s);
//                assert(rew[s]==1);
//            }

//        }



    }

    Matrix getBasis3(){
        //Matrix F5 = getBasis2();
        F4.Resize(n_states, 10);

        for (int s = 0; s < n_states; s++) {

            // Features
            // NB: the first index ( of 3, second dim) is not used
            std::vector<std::vector<int> > features(3, std::vector<int>(3,0));
            std::vector<std::vector<int> > singlet_count(9,std::vector<int>(3,0));

            std::vector<std::vector<int> > diversity(9,std::vector<int>(3,0));

             std::vector<std::vector<int> > opportunity(9,std::vector<int>(3,0));

        // Looping through the row
        for(int row=0; row<3; row++){

            std::vector<TTT_CASE> board(3);

            for(int col = 0; col<3; col++){

                //board[col] = sol[s][col+row*3];
                board[col].player = ttt_int2player(sol[s][col+row*3]);
                board[col].case_id = col+row*3;


            }

            update(board,singlet_count,features,diversity,opportunity,s);


        }
        assert(features[1][1] <= 2 && features[1][2]<=2);

        // Looping through the columns

        for(int col=0; col<3; col++){

            std::vector<TTT_CASE> board(3);
            for(int row=0; row<3; row++){

                //board[row] = sol[s][row+col*3];
                board[row].case_id = row*3+col;
                board[row].player = ttt_int2player(sol[s][row*3+col]);


            }

            update(board,singlet_count,features,diversity,opportunity,s);
        }

        // Looping through the main diagonals

        std::vector<TTT_CASE> board1,board2;

        for(int row =0; row<3; row++){

            for(int col = 0; col<3; col++){

                // First diagonal
                if(row == col){


                    TTT_CASE tmp;
                    tmp.case_id = row+col*3;
                    tmp.player = ttt_int2player(sol[s][row+col*3]);
                    board1.push_back(tmp);
                }

                // Second diagonal
                if(row == 2-col){

                    TTT_CASE tmp;
                    tmp.case_id = col+row*3;
                    tmp.player = ttt_int2player(sol[s][col+row*3]);
                    board2.push_back(tmp);

                }
            }
        }
        update(board1,singlet_count,features, diversity,opportunity,s);
        update(board2,singlet_count,features, diversity,opportunity,s);

        // Computing number of crosspoint

        std::vector<int> num_cross(3,0);

        for(int i=0; i< 9; i++){

            if(singlet_count[i][TTT_PO] >= 2)
                num_cross[TTT_PO] += 1;
            if(singlet_count[i][TTT_PX] >= 2)
                num_cross[TTT_PX] += 1;


        }

        // Computing number of diversity

        std::vector<int> num_diversity(3,0);

        for(int i=0; i< 9; i++){

            if(diversity[i][TTT_PO] >= 2)
                num_diversity[TTT_PO] += 1;
            if(diversity[i][TTT_PX] >= 2)
                num_diversity[TTT_PX] += 1;


        }


        // Computing number of case X or O belonging to two different
        // doublets

        std::vector<int> num_oppor(3,0);

        for(int i=0; i< 9; i++){

            if(opportunity[i][TTT_PO] >= 2)
                num_oppor[TTT_PO] += 1;
            if(opportunity[i][TTT_PX] >= 2)
                num_oppor[TTT_PX] += 1;


        }

        // Computing the features

        int num = 0;

        for(int player2=1; player2< 3; player2++){

            F4(s,num) = features[TTT_SINGLET][player2];
            num++;
            F4(s, num) = features[TTT_DOUBLET][player2];
            num++;
            F4(s, num) = features[TTT_TRIPLET][player2];
            num++;
            F4(s,num) = num_cross[player2];
            num++;
            F4(s,num) = num_diversity[player2];
            num++;
            //F4(s,num) = num_oppor[player2];
            //num++;
        }

        }



//        std::cerr<<"Testing features "<<std::endl;
//        for(int s=0; s<n_states; s++){

//            for(int i=0; i<8; i++){

//                if(i <2)
//                    assert(F4(s,i) == F5(s,i));
//                if(i==4 || i == 5)
//                    assert(F4(s,i) == F5(s,i-2));
//                std::cerr<<F4(s,i)<<"\t";
//            }
//            std::cerr<<std::endl;
//        }

    return F4;
    }

	void compute_basis_4(int num = 4){

		v_basis.reserve(n_states);
		VectorXd tmp=Eigen::VectorXd::Ones(num);
        //assert(F4.Columns() == 8);
		for(int i=0; i< n_states; i++){

            for(int j=0; j<F4.Columns();j++){

				tmp(j) = F4(i,j);
			}
			v_basis.push_back(tmp);
		}
        scale(v_basis,-1,1);

		v_basis_bis = std::vector<VectorXmp>(v_basis.size());

		for(int i=0; i<v_basis.size();i++){

            v_basis_bis[i] = v_basis[i].cast<mpreal>();
		}

	}

	void compute_basis_9(int num = 13){

		v_basis.reserve(n_states);
		VectorXd tmp=Eigen::VectorXd::Ones(num);
		for(int i=0; i< n_states; i++){

            for(int j = 0; j< 9; j++){
                //tmp(j) = (sol[i][j]-0.9)/2.0;
                int er =0;
                if(sol[i][j] == 0){
                    er = 0;
                }
                else if(sol[i][j] == 1){
                    er = 1;
                }
                else {
                    er = -1;
                }
                //tmp(j) = sol[i][j];
                tmp(j) = er;

            }

            for(int j=0; j<9; j++){

                if(sol[i][j] == 1){
                    tmp(j) = 1;
                }
                else {
                    tmp(j) = 0;
                }
            }

            for(int j=9; j<18; j++){

                if(sol[i][j-9] == 2){
                    tmp(j) = -1;
                }
                else {
                    tmp(j) = 0;
                }
            }

//            for(int j=0; j<6;j++){
//                tmp(j+9) = 0;
//            }

//            for(int j=0; j<9;j++){

//                if(j==0 || j==2 || j==6 || j==8){

//                    if(sol[i][j] == 1){
//                        tmp(0+9) += 1;
//                    }
//                    else if(sol[i][j] == 2){
//                        tmp(0+9+3) += 1;
//                        //tmp(0+9+1) += 1;
//                    }
//                    //continue;
//                }


//                else if(j==4){
//                    if(sol[i][j] == 1){
//                        tmp(1+9) += 1;
//                    }
//                    else if(sol[i][j] == 2){
//                        tmp(1+3+9) += 1;
//                    }

//                }

//                else {
//                    if(sol[i][j] == 1){
//                        tmp(2+9) += 1;
//                    }
//                    else if(sol[i][j] == 2){
//                        tmp(2+3+9) += 1;
//                    }

//                }
//            }
//            int idx2 = 1*2+9;
            int idx2 = 9;

//            for(int j=0; j<9; j++){

//                for(int k=0; k<9;k++){

//                    tmp(idx2) = tmp(j)*tmp(k);
//                    idx2++;
//                }
//            }

            if(num>9){

                //assert(F4.Columns() == 8);

            for(int j = 0; j<F4.Columns(); j++){

                tmp(j+idx2)= F4(i,j);
			}

            int idx = idx2+F4.Columns();

//            for(int e=2; e<6;e++){


//                for(int i=0; i< 9+F4.Columns(); i++){

//                    tmp(idx) = std::pow(tmp(i),e);
//                    idx++;
//                }
//            }

            //for(int j=0; j<6; j++){
              //  tmp(j+9+6) = F4(i,j)*F4(i,j);
            //}
            //int idx = 9+6;

            for(int j=0; j<F4.Columns();j++){

                for(int k=0; k<F4.Columns(); k++){

                    tmp(idx) = F4(i,j)*F4(i,k);
                    idx++;
                }
            }

            //tmp(idx) = F4(i,2);
            //idx++;
            //tmp(idx) = F4(i,8);

//            for(int j=0; j<9+F4.Columns();j++){

//                for(int k=0; k<9+F4.Columns(); k++){

//                    tmp(idx) = tmp(j)*tmp(k);
//                    idx++;
//                }
//            }
			}

			v_basis.push_back(tmp);
		}

        scale(v_basis,-1,1);


		v_basis_bis = std::vector<VectorXmp>(v_basis.size());

		for(int i=0; i<v_basis.size();i++){

            v_basis_bis[i] = v_basis[i].cast<mpreal>();
		}

	}


	virtual VectorXd& get_basis(int i_ep,int i_ns, DEMO_MODE demo_st,
				int s, int a =  -1, int s_next =  -1, bool q_features_available_ = false){

		if(s_next != -1 && a != -1){

			return v_basis[s_next];
		}
		else if (s_next == -1 && a != -1) {
			if(transition[s].size()>0){
				return v_basis[transition[s][a]];
			}
			else {
				return v_basis[s];
			}


		}
		else{
			return v_basis[s];
		}

		return v_basis[s];
	}


	virtual VectorXmp& get_basis(int s, int a =  -1, int s_next =  -1){

		if(s_next != -1 && a != -1){

			return v_basis_bis[s_next];
		}
		else if (s_next == -1 && a != -1) {
			if(transition[s].size()>0){
				return v_basis_bis[transition[s][a]];
			}
			else {
				return v_basis_bis[s];
			}


		}
		else{
			return v_basis_bis[s];
		}

		return v_basis_bis[s];
	}

	///  Get the Basis : Here a is an index of transition[s]


	/*Matrix get_basis(int s, int a = -1, int s_next = -1) {
		assert(s < n_states);
		assert((a == -1 || a < get_n_actions(s)));
		assert((s_next == -1 || s_next < n_states));
		Matrix tmp(n_features, 1);
		int s_actu = s;

		if (s_next != -1 && a != -1) {
			s_actu = s_next;
		}

		if (s_next != -1 && a != -1) {
			s_actu = s_next;

			//tmp(0,0) = basis(s_next, 0) + basis(transition[s][a], 0);
			//tmp(1, 0) = basis(s_next, 1) + basis(transition[s][a], 1);
			//tmp(2, 0) = basis(s_next, 2) + basis(transition[s][a], 2);
			//tmp(3, 0) = basis(s_next,3) + basis(transition[s][a], 3);

			tmp(0, 0) = basis(s_next, 0);
			tmp(1, 0) = basis(s_next, 1);
			tmp(2, 0) = basis(s_next, 2);
			tmp(3, 0) = basis(s_next, 3);
		} else if (s_next == -1 && a != -1) {

			/// This possibility should never happen ??
			/// exiting if so
			//std::cerr<<"Make sure it does not happen"<<std::endl;
			//exit(0);
			int s2 = transition[s][a];

			//tmp(0,0) = basis(s2, 0);
			//tmp(1, 0) = basis(s2, 1);
			//tmp(2, 0) = basis(s2, 2);
			//tmp(3, 0) = basis(s2,3) ;
			tmp(0, 0) = 0;
			tmp(1, 0) = 0;
			tmp(2, 0) = 0;
			tmp(3, 0) = 0;
			for (int i = 0; i < (int) transition[s2].size(); i++) {

				//tmp(0,0) += basis(transition[s2][i], 0) + basis(s2, 0);
				//tmp(1, 0) += basis(transition[s2][i], 1) + basis(s2, 1);
				//tmp(2, 0) += basis(transition[s2][i], 2) + basis(s2, 2);
				//tmp(3, 0) += basis(transition[s2][i],3) + basis(s2, 3);

				tmp(0, 0) += basis(transition[s2][i], 0);
				tmp(1, 0) += basis(transition[s2][i], 1);
				tmp(2, 0) += basis(transition[s2][i], 2);
				tmp(3, 0) += basis(transition[s2][i], 3);

			}

			if (transition[s2].size() != 0) {
				tmp(0, 0) /= (int) transition[s2].size();
				tmp(1, 0) /= (int) transition[s2].size();
				tmp(2, 0) /= (int) transition[s2].size();
				tmp(3, 0) /= (int) transition[s2].size();
			}

			else {

				tmp(0, 0) = basis(s, 0);
				tmp(1, 0) = basis(s, 1);
				tmp(2, 0) = basis(s, 2);
				tmp(3, 0) = basis(s, 3);
			}

		}

		else {

			tmp(0, 0) = basis(s, 0);
			tmp(1, 0) = basis(s, 1);
			tmp(2, 0) = basis(s, 2);
			tmp(3, 0) = basis(s, 3);
		}

		//for(int i=4;i<13;i++)
		 //{
		 //tmp(i-4,0)=sol[s_actu][i-4];
		 //}

		return tmp;

	}*/

	int get_n_actions(int s = 0) {
		return transition[s].size();
	}

	int get_index_state(int s, int a)
	 {
	 return std::distance(transition[s].begin(),std::find(transition[s].begin(),transition[s].end(),a));

	 }

	 int get_state_index(int s, int a)
	 {
	 //std::cerr<< "T size2 = "<<transition.size()<<"\t";
	 return transition[s][a];
	 }

	void display(int s2 = 0) {
		int count = 0;
		for (int s = s2; s < n_states; s++) {
			if (count == 1)
				break;
            std::cerr << " state " << s2 << std::endl;
			for (int k = 0; k < 9; k++) {
				if (k % 3 == 0)
                    std::cerr << std::endl;
                std::cerr << sol[s][k] << "  ";

			}
            std::cerr << std::endl;
			/*for(set<int>::iterator it=succ[s].begin();it!=succ[s].end();it++)
			 {
			 for(int k=0;k<9;k++)
			 {
			 if(k%3==0)
			 cout<<endl;
			 cout<<sol[*it][k]<<"  ";

			 }

			 }*/
			count++;
		}
	}

	virtual ~AppTTT() {

	}
	;
};

#endif /* APPTTT_H_ */
