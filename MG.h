/*
 * MG.h
 *
 *  Created on: 8 Sep 2012
 *      Author: yedtoss
 *
 *      Markov Games Framework
 */

#ifndef MG_H_
#define MG_H_

#include <vector>
#include <set>
//using namespace std;
typedef std::vector<int> vi;
typedef std::vector<std::vector<int> > vii;

class MG {
public:
	vi reward, player;
	std::vector<std::set<int> > succ;
	int n_states;
	MG();
	MG(vi r, std::vector<std::set<int> > succ_, vi player_) {
		n_states = r.size();
		reward = r;
		succ = succ_;
		player = player_;
	}
	int getNStates() {
		return n_states;
	}
	int getNActions() {
		return 0;
	}
};

#endif /* MG_H_ */
