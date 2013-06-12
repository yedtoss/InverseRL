/*
 * AppEnvironment.h
 *
 *  Created on: 7 Sep 2012
 *      Author: yedtoss
 */

#ifndef APPENVIRONMENT_H_
#define APPENVIRONMENT_H_

#include <Eigen/Dense>
#include <iostream>

#include "MyTypes.h"
#include "LinearApproximator.h"
//#include "NNApproximator.h"
#include "FNNApproximator.h"

class AppEnvironment {
public:

	std::vector<std::vector<std::vector<std::vector<VectorXd> > > > demo_basis,
			demo_q_basis;
	std::vector<VectorXd> hash_basis;
	std::vector<std::vector<int> > hash_tr;
	std::vector<std::vector<std::vector<std::vector<std::vector<int> > > > > tr;
	int hash_limit, hash_current, hash_tr_limit, hash_tr_current;
	bool q_features_available;
	bool isLinear;
	double reward;
	int state, n_features, n_states, n_actions;
	bool MultipleNN;
	Approximator *app, *rapp, *qapp, *vapp;
	App type;
	VectorXmp rien;
    bool use_kernel;
    MatrixXmp kernel_matrix_bis;
    MatrixXd kernel_matrix;
    MatrixXmp  kernel_components;
    VectorXmp kernel_mean;
    MatrixXmp tmp4;
    Demo_C demo_comp;
    double kernel_sig;
    int features_type, n_points;

	/// In the following code
	//AppEnvironment(){

	//}

	virtual std::string Name() {
		return "Base Class for approximate environment";
	}
	virtual double get_reward() {
		return reward;
	}

	virtual int get_state() {
		return state;
	}

	/*virtual bool act(int action) {
		return 0;
	}*/

	virtual bool act(int action, bool autom = true, int s2 = -1){
		return 0;
	}

	virtual void reset() {

	}
	virtual int get_n_features() {
		return n_features;
	}
	virtual int get_num_training() {

		return app->get_num_training();
	}

	virtual int get_r_num_training() {

		return rapp->get_num_training();
	}

	virtual int get_q_num_training() {

		return qapp->get_num_training();
	}

	virtual int get_v_num_training() {

		return vapp->get_num_training();
	}

	virtual int get_n_states() {
		return n_states;
	}
	virtual int get_n_actions(int s = 0) {
		assert(s<n_states);
		return n_actions;
	}

	/// a is index of a state and we are looking for it's id in the vector

	virtual int get_index_state(int s, int a) {

		assert(s < n_states);

		return a;

	}

	virtual int get_state_index(int s, int a) {

		return a;
	}

    virtual void compute_demo(Demonstration &D_, int max_points =INT_MAX)
    {
        demo_comp = compress(D_);

        n_points = std::min(max_points, demo_comp.size());

        std::cout<<"Number of stats observed "<<demo_comp.size()<<std::endl;

    }

    virtual void compute_kernel_matrix()
    {
        //kernel_matrix_bis = MatrixXmp(n_features,n_points);


        // Making sure that use_kernel is not already activated
        bool tmp = use_kernel;
        use_kernel = false;
        int demo_s = demo_comp.size();
        kernel_matrix_bis = MatrixXmp(n_features,demo_s);
        std::cout<<"Computing kernel matrix"<<std::endl;

        for(int nep=0; nep<demo_s; nep++)
        {
            kernel_matrix_bis.col(nep) = get_basis(demo_comp[nep].s);
        }

        // Let's compute the kernel data for the demonstration
        std::cout<<"Computing kernel data"<<std::endl;
        MatrixXmp kernel_data = MatrixXmp(demo_s,demo_s);
        for(int nep=0; nep<demo_s; nep++)
        {
            //kernel_data.row(nep) = (-kernel_sig*((kernel_matrix_bis.colwise()-get_basis(demo_comp[nep].s)).colwise().norm())).array().exp().matrix();
            tmp4 = (((kernel_matrix_bis.colwise()-get_basis(demo_comp[nep].s))));
            tmp4.resize(1,tmp4.size());
            kernel_data.row(nep) = tmp4;

        }
        std::cout<<"Computing kernel mean"<<std::endl;
        kernel_mean = (kernel_data.colwise().mean()).transpose();
        kernel_data.rowwise() -= kernel_mean.transpose();

        std::cout<<"Computing SVD of kernel data"<<std::endl;

        Eigen::JacobiSVD<MatrixXmp> svd(kernel_data,Eigen::ComputeThinV);

        std::cout<<"Computing principal components of kernel data"<<std::endl;


        /// TO BE FIXED
        //kernel_components = ((svd.matrixV().block(0,0,demo_s,n_points)).cwiseQuotient((svd.singularValues().head(n_points).array()+EPS2).matrix().rowwise().replicate(demo_s).transpose()))*sqrt(demo_s);
        std::cout<<"PCA terminated"<<std::endl;

        assert(kernel_components.rows() == demo_s);
        assert(kernel_components.cols() == n_points);

        // Now we don't need anymore kernel_data

        // Not used
        //kernel_matrix = kernel_matrix_bis.cast<double>();
        std::cerr<<"kernel matrix"<<kernel_matrix_bis<<std::endl;

        use_kernel = tmp;

    }

	virtual VectorXd& get_v_basis(int i_ep, int i_ns, DEMO_MODE demo_st, int s,
			int a = -1, int s_next = -1) {

		return get_basis(i_ep, i_ns, demo_st, s, a, s_next);

	}

	virtual VectorXd& get_q_basis(int i_ep, int i_ns, DEMO_MODE demo_st, int s,
			int a = -1, int s_next = -1) {

		return get_basis(i_ep, i_ns, demo_st, s, a, s_next,
				q_features_available);

	}

	virtual VectorXd& get_r_basis(int i_ep, int i_ns, DEMO_MODE demo_st, int s,
			int a = -1, int s_next = -1) {

		return get_basis(i_ep, i_ns, demo_st, s, a, s_next);
	}


	virtual VectorXd& get_basis(int i_ep, int i_ns, DEMO_MODE demo_st, int s,
			int a = -1, int s_next = -1, bool q_features_available_ = false) {

		if (q_features_available_) {

			if (demo_st == DEMO_CURRENT) {
				return demo_q_basis[i_ep][i_ns][0][0];
			} else if (demo_st == DEMO_NEXT) {
				return demo_q_basis[i_ep][i_ns][1][0];
			}

			else if (demo_st == DEMO_ACT_CURRENT) {

				return demo_q_basis[i_ep][i_ns][0][a - 1];
			}

			else if (demo_st == DEMO_ACT_NEXT) {
				return demo_q_basis[i_ep][i_ns][1][a - 1];
			}

			return basis(s, a, s_next);
		}

		if (demo_st == DEMO_CURRENT) {
			return demo_basis[i_ep][i_ns][0][0];
		} else if (demo_st == DEMO_NEXT) {
			return demo_basis[i_ep][i_ns][1][0];
		}

		else if (demo_st == DEMO_ACT_CURRENT) {

			return demo_basis[i_ep][i_ns][0][a - 1];
		}

		else if (demo_st == DEMO_ACT_NEXT) {
			return demo_basis[i_ep][i_ns][1][a - 1];
		}

		//return VectorXd(1);
		return basis(s, a, s_next);
		return hash_basis[hash_current];
	}

	virtual VectorXmp& get_v_basis(int s,int a = -1, int s_next = -1) {

		return get_basis(s, a, s_next);

	}

	virtual VectorXmp& get_q_basis(int s,int a = -1, int s_next = -1) {

		return get_basis( s, a, s_next);

	}

	virtual VectorXmp& get_r_basis(int s,int a = -1, int s_next = -1) {

		return get_basis(s, a, s_next);
	}

	virtual VectorXmp& get_basis(int s,int a = -1, int s_next = -1) {

			    return rien;

    }


	virtual VectorXd& basis(int s, int a = -1, int s_next = -1) {

		return hash_basis[hash_current];

	}

	virtual double get_r(VectorXd& w_, int i_ep, int i_ns, DEMO_MODE demo_st,
			int s, int a = -1, int s_next = -1) {
		VectorXd& tmp = get_r_basis(i_ep, i_ns, demo_st, s, a, s_next);
		return rapp->predict(tmp, w_);

	}
	virtual double get_v(VectorXd& w_, int i_ep, int i_ns, DEMO_MODE demo_st,
			int s, int a = -1, int s_next = -1) {
		VectorXd& tmp = get_v_basis(i_ep, i_ns, demo_st, s, a, s_next);
		return vapp->predict(tmp, w_);

	}

	virtual double get_q(VectorXd& w_, int i_ep, int i_ns, DEMO_MODE demo_st,
			int s, int a = -1, int s_next = -1) {

		//if(q_features_available){

		assert(s!= -1 && a!= -1);

		VectorXd& tmp = get_q_basis(i_ep, i_ns, demo_st, s, a, s_next);
		if(MultipleNN){

			return qapp->predict2(tmp, w_)[a];

		}

		else {
			return qapp->predict(tmp, w_);
		}




		//}

		/*double rep2 = 0;

		 if(a!=-1 && s_next==-1){

		 std::vector<int>& t = transition(i_ep,i_ns, demo_st, s, a);
		 //VectorXd tmp;
		 for(int i = 0; i< t.size(); i++){

		 VectorXd& tmp = get_q_basis(i_ep,i_ns, demo_st, s, a, t[i]);
		 rep2 += qapp->predict(w_,tmp);
		 }

		 if(t.size() == 0){
		 VectorXd& tmp = get_q_basis(i_ep,i_ns, demo_st,s);
		 rep2 = qapp->predict(w_,tmp);
		 }
		 else{
		 rep2 /= t.size();
		 }
		 }
		 else{
		 VectorXd& tmp = get_v_basis(i_ep,i_ns, demo_st,s, a, s_next);
		 rep2 = vapp->predict(tmp, w_);
		 }


		 return rep2;*/

	}


	virtual VectorXd get_q2(VectorXd& w_, int i_ep, int i_ns, DEMO_MODE demo_st,
				int s, int a = -1, int s_next = -1) {

			//if(q_features_available){

			VectorXd& tmp = get_q_basis(i_ep, i_ns, demo_st, s, a, s_next);
			return qapp->predict2(tmp, w_);

	}




	/*virtual std::vector<int>& transition(int s,int a,int i_ep,int i_ns, DEMO_MODE demo_st){

	 if(demo_st == DEMO_CURRENT){
	 return tr[i_ep][i_ns][0][0];
	 }
	 else if(demo_st == DEMO_NEXT){
	 return tr[i_ep][i_ns][1][0];
	 }

	 else if(demo_st == DEMO_ACT_CURRENT){

	 return tr[i_ep][i_ns][0][a-1];
	 }

	 else if (demo_st == DEMO_ACT_NEXT){
	 return tr[i_ep][i_ns][1][a-1];
	 }

	 return trans(s, a);


	 }
	 virtual std::vector<int>& trans(int s, int a){

	 return tr[hash_tr_current];


	 }*/

	/*
	 *  Is the action a valid at state s?
	 */

	virtual bool isValid(int s,int a){

			return true;
		}

	/*
	 *   Return \max_b Q(s,b)
	 */

	virtual double get_qmax(VectorXd& w_, int i_ep, int i_ns, DEMO_MODE demo_st,
				int s, int a = -1, int s_next = -1) {

		int num_t = get_n_actions(s);
		assert(a== -1 && s_next==-1);
		double cur, best= -5000;

		if(num_t<=0){
			return 0;
		}

		VectorXd all_q=VectorXd::Zero(num_t);

		if(MultipleNN){

			all_q = get_q2(w_, i_ep, i_ns, demo_st, s, 0);
		}
		else {

			for (int i = 0; i < num_t; i++) {
						if(!isValid(s,i))
							continue;
						all_q(i) = get_q(w_, i_ep, i_ns, demo_st, s, i);
					}

		}


		for (int i = 0; i < num_t; i++) {

			//cur = get_q(w_, i_ep, i_ns, demo_st, s, i);
			cur = all_q(i);
			if(!isValid(s,i))
				continue;
			if (cur > best) {
				best = cur;
			}
		}

		return best;
	}

	/*
	 * Return \argmax_b Q(s,b)
	 */

	virtual int policy_act(VectorXd& pi, int s, bool deterministic = true,
				int i_ep = -1, int i_ns = -1, DEMO_MODE demo_st = DEMO_NONE) {




			int a = 0;
			//std::vector<int>& t=transition(s,-1,i_ep, i_ns, demo_st);
			int num_t = get_n_actions(s);

			if (num_t <= 0)
				return 0;
			VectorXd all_q = VectorXd::Zero(num_t);

			if(MultipleNN){

				all_q = get_q2(pi, i_ep, i_ns, demo_st, s, 0);
			}
			else {

				for (int i = 0; i < num_t; i++) {
								if(!isValid(s,i))
									continue;
								all_q(i) = get_q(pi, i_ep, i_ns, demo_st, s, i);
							}

			}



			if (get_n_actions(s) > 0)
				a = rand() % num_t;

			/// Returning the action with the greater Q
			/// If there are multiple such action return one at random

			if (deterministic) {
				double best = -5000, cur = -5000;
				//cur = get_q(pi, i_ep, i_ns, demo_st, s, 0);
				//best = cur;
				bool isEveryThingEqual = true;
				a = 0;
				/// num: Number of actions with the greatest Q
				int offset = 0, num = 1;

				/// The element from offset to the last element of l are the action
				/// with the greatest Q
				std::vector<int> l;
				l.reserve(num_t + 2);
				//l.push_back(a);

				/// We don't want to choose any action from the beginning anymore

				best = -5000;
				num = 0;

				//for (int i = 1; i < num_t; i++) {
				for (int i = 0; i < num_t; i++) {


					//cur = get_q(pi, i_ep, i_ns, demo_st, s, i);
					cur = all_q(i);
					if(!isValid(s,i))
						continue;

					if (cur > best && isValid(s,i)) {

						best = cur;
						a = i;
						isEveryThingEqual = false;
						offset = l.size();
						l.push_back(i);
						num = 1;

                    } else if (std::abs(cur- best)<1e-15 && isValid(s,i)&&(MultipleNN ||
							(get_q_basis(-1, -1, DEMO_ACT_NEXT, s,a)-get_q_basis(-1, -1, DEMO_ACT_NEXT, s,i)).squaredNorm()>1e-10)) {
					//else if(cur == best){
						l.push_back(i);
						num++;
					}

				}

				if (num > 1) {

					return l[offset + rand() % num];
					//return 0;

				} else {
					return a;
				}

			}

			/// Choosing an action by following a non-deterministic policy
			/// Element with greater Q will have more chance to be pick
			/// So each action will be pick according to his Q
			else {

				// Not yet implemented

				return 0;
			}

		}

    /// Return the greedy policy for all states
    virtual std::vector<int> greedy_pol(VectorXd& pi){

        int num3 = get_n_states();
        assert(num3 > 0);
        std::vector<int> pol(num3,0);

        for(int s=0; s<num3;s++){

            pol[s] = policy_act(pi,s);
        }

        return pol;


    }

	virtual ~AppEnvironment() {

	}
};

#endif /* APPENVIRONMENT_H_ */
