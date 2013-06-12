/*
 * FNNApproximator.h
 *
 *  Created on: 11 Sep 2012
 *      Author: yedtoss
 */

#ifndef FNNAPPROXIMATOR_H_
#define FNNAPPROXIMATOR_H_

#include "Approximator.h"

class FNNApproximator : public Approximator{
public:
	int hidden_neurons_number, outputs_number, num_input;
	vi network,t,t2,network2;
	int L;
    MatrixXd b,w;
    VectorXd b2;
    IRL_ACTIVATION act;
    int offset;
    int num_net;
    static const double coeff =2.0/3.0;
    static const double fact = 1.7159;
	FNNApproximator(int num_input_,IRL_ACTIVATION act_ = ACT_LOGSIG,int num_net_ = 1){
		//network.resize(3);
		num_input = num_input_;
		network2.resize(3,num_input);
		network2[0] = num_input;
		network2[1] = num_input;
		network2[2] = 1;
		offset = 0;
		num_net = num_net_;
		//this->(network2,act_);
		init(network2,act_);
		//num_training = 0;
		//num_training+=36;

		//this->FNNApproximator(network_,act_);
		/*num_input = num_input_;
		outputs_number = 1;
		hidden_neurons_number = num_input;
		L = 2;
		t.resize(L);
		network[0] = num_input;
		network[1] = num_input;
		network[2] = 1;
		t[0] = 0;
		t[1] = num_input*hidden_neurons_number;
		num_training = t[0] + t[1] + hidden_neurons_number*outputs_number;
		act = act_;*/

	}
	FNNApproximator(vi& network_,IRL_ACTIVATION act_ = ACT_LOGSIG,int num_net_ = 1){
		assert(network_.size()>=2);
		network = network_;
		num_input = network[0];
		std::cout<<"in "<<num_input<<std::endl<<std::endl;
		outputs_number = network[network.size()-1];
		hidden_neurons_number = network.size() - 2;
		num_training = 0;
		L = network.size()-1;

		// Starting indice of the weight in layer i
		t.resize(L,0);
		t=vi(L,0);
		// Starting indices of the bias in layer i
		t2.resize(L,0);
		t2 = vi(L,0);
		for(int i = 1; i <L;i++){

			t2[i-1] = t[i-1]+ network[i-1]*network[i];
			t[i] = t2[i-1] + network[i];

			//t[i]=network[i-1]*network[i];

		}
		t2[L-1] = t[L-1] + network[L-1]*network[L];
		for(int i=0; i< L;i++){

			//num_training+=network[i]*network[i+1];
			num_training+=((network[i] +1)*network[i+1]);

		}

		std::cout<<"num t "<<num_training<<std::endl;

		//std::cout<<num_training<<std::endl;
		//exit(0);

		act = act_;
		offset = 0;
		num_net = num_net_;
		//act = ACT_LOGSIG;



	}

	void init(vi& network_,IRL_ACTIVATION act_ = ACT_LOGSIG,int num_net_ = 1){

		assert(network_.size()>=2);
				network = network_;
				num_input = network[0];
				std::cout<<"in "<<num_input<<std::endl<<std::endl;
				outputs_number = network[network.size()-1];
				hidden_neurons_number = network.size() - 2;
				num_training = 0;
				L = network.size()-1;

				// Starting indice of the weight in layer i
				t.resize(L,0);
				t=vi(L,0);
				// Starting indices of the bias in layer i
				t2.resize(L,0);
				t2 = vi(L,0);
				for(int i = 1; i <L;i++){

					t2[i-1] = t[i-1]+ network[i-1]*network[i];
					t[i] = t2[i-1] + network[i];

					//t[i]=network[i-1]*network[i];

				}
				t2[L-1] = t[L-1] + network[L-1]*network[L];
				for(int i=0; i< L;i++){

					//num_training+=network[i]*network[i+1];
					num_training+=((network[i] +1)*network[i+1]);

				}

				std::cout<<"num t "<<num_training<<std::endl;

				//std::cout<<num_training<<std::endl;
				//exit(0);

				act = act_;
				offset = 0;
				num_net = num_net_;
				//act = ACT_LOGSIG;

	}


	/// Predicting with bias

	virtual double predict(VectorXd& input, VectorXd& weight) {

		b2 = input;
		Eigen::Map<Eigen::MatrixXd> B2(weight.data(), 2, 2);
		Eigen::Map<Eigen::VectorXd> bias(weight.data(), 2);

		for (int i = 0; i < L; i++) {
			new (&B2) Eigen::Map<Eigen::MatrixXd>(weight.data() + t[i]+offset, network[i + 1],
					network[i]);
			new (&bias) Eigen::Map<VectorXd>(weight.data()+t2[i]+offset,network[i + 1]);

			b2 = B2 * b2 + bias;
			//std::cout<<"b2 "<<b2.rows()<<"bias"<<bias.rows();

			//if(i < L-1){
			//if(i > 0){
			for (int j = 0; j < b2.rows(); j++) {

				if (act == ACT_TANH) {
					b2(j) = fact*std::tanh(coeff*(double)b2(j));
				} else {
					b2(j) = 1.0 / (1.0 + std::exp((double)-b2(j)));
				}
			}
			//}

		}
		//std::cout<<b2(0)<<" ";
		return b2(0);

	}

	virtual VectorXd predict2(VectorXd& input, VectorXd& weight) {


			VectorXd rep=VectorXd::Zero(num_net);
			for(int i=0; i< num_net;i++){

				offset = i*num_training;

				rep(i)=predict(input,weight);

			}

			return rep;
		}


	/*virtual double predict(VectorXd& input, VectorXd& weight){

		b = input.transpose();
		Eigen::Map<Eigen::MatrixXd> B2(weight.data(),2,2);

		for(int i=0; i< L;i++){
			//std::cout<<b.cols()
			//		<<" ";
			new (&B2) Eigen::Map<MatrixXd>(
					weight.data()+t[i],network[i],network[i+1]);

			//std::cout<<B2.rows()<<" "<<B2.cols()<<" ";

			//b= b*weight.block(t[i],0,network[i],network[i+1]);
			b = b*B2;
			for(int j = 0; j< b.cols();j++){

				if(act == ACT_TANH){
					b(0,j) = std::tanh(b(0,j));
				}
				else{
					b(0,j) = 1.0/(1.0+std::exp(-b(0,j)));
				}


			}
		}
		//std::cout<<b(0,0)<<" ";
		//std::cout<<std::endl;

		return b(0,0);
	}*/

	virtual int get_num_training(){

			return num_training*num_net;
		}

	virtual ~FNNApproximator(){

	}
};

#endif /* FNNAPPROXIMATOR_H_ */
