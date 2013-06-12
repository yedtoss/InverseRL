/*
 * VIAMG.h
 *
 *  Created on: 9 Sep 2012
 *      Author: yedtoss
 */

#ifndef VIAMG_H_
#define VIAMG_H_

#include "Matrix.h"
#include "Vector.h"
//#include "real.h"
#include <vector>
#include <set>
//using namespace std;
typedef std::vector<int> vi;
typedef std::vector<std::vector<int> > vii;

class VIAMG {
public:

	MG* mdp;
	double gamma;
	int n_states;
	int n_actions;
	Vector V;
	vi player, reward, policy;


	VIAMG(MG* mdp_, vi reward_, double gamma_) {
		mdp = mdp_;
		reward = reward_;
		this->gamma = gamma_;
		n_actions = mdp->getNActions();
		n_states = mdp->getNStates();
		V.Resize(n_states);
		player = mdp->player;
		policy.resize(n_states, 0);
		for (int s = 0; s < n_states; s++)
			V(s) = 0;

	}

	void ComputeStateValues(double threshold, int max_iter = -1) {
		double delta = 0;

		do {
			delta = 0;
			for (int s = 0; s < n_states; s++) {

				double v = V(s);

				std::set<int>::iterator it = mdp->succ[s].begin();
				if (it != mdp->succ[s].end()) {
					V(s) = reward[*it] + gamma * V(*it);
					policy[s] = *it;
					it++;
					//if(s==78)
					//cout<<" rew "<<reward[*it]<<"g "<<gamma<<"V "<<V(*it)<<endl;
				}
				for (it = mdp->succ[s].begin(); it != mdp->succ[s].end();
						it++) {
					double res = reward[*it] + gamma * V(*it);
					if (player[s] == 0 && (res > V(s))) {
						V(s) = res;

						policy[s] = *it;

					}
					if (player[s] == 1 && (res < V(s))) {

						V(s) = res;
						policy[s] = *it;
					}
				}
				//delta+=std::abs(v-V(s));
				delta = std::max(delta, std::abs(v - V(s)));
			}
			max_iter--;
			delta = -2;
		} while (delta < threshold && max_iter != 0);

		/*for(int s=0;s<n_states;s++)
		 {
		 double sm=0;
		 int ind=0;
		 if(mdp->succ[s].size()>0)
		 {
		 ind=*(mdp->succ[s].begin());
		 sm=V(ind);

		 }

		 for(std::set<int>::iterator it=mdp->succ[s].begin(); it!=mdp->succ[s].end();it++)
		 {

		 if(player[s]==0 && (V(*it)>sm  ))
		 {
		 sm=V(*it);
		 ind=*it;
		 }
		 if(player[s]==1 && (V(*it)<sm  ) )
		 {
		 sm=V(*it);
		 ind=*it;
		 }

		 }
		 policy[s]=ind;

		 }*/
	}

	vi getPolicy() {
		vi n_p = std::vector<int>(n_states, 0);
		for (int s = 0; s < n_states; s++) {
			(n_p)[s] = policy[s];
		}
		return n_p;
	}

};

#endif /* VIAMG_H_ */
