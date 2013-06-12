/*
 * NNApproximator.h
 *
 *  Created on: 8 Sep 2012
 *      Author: yedtoss
 */

#ifndef NNAPPROXIMATOR_H_
#define NNAPPROXIMATOR_H_

#include "../tinyxml/tinyxml.h"
#include "opennn.h"
#include "Approximator.h"


class NNApproximator : public Approximator {
public:
	int hidden_neurons_number, outputs_number, num_input;
	OpenNN::MultilayerPerceptron *perc;
	int num_net;
	OpenNN::DataSet* dset;

	std::vector<OpenNN::DataSet> vdset;
	std::vector<int> num_vdset;
	vi Q_act;

	// Two Layer construction
	NNApproximator(int num_input_,int hidden_neurons_number_,int outputs_number_,int num_net_ = 1) {

		num_input = num_input_;
		hidden_neurons_number = hidden_neurons_number_;
		outputs_number = outputs_number_;
		perc = new OpenNN::MultilayerPerceptron(num_input,hidden_neurons_number,
				outputs_number);
		num_training = perc->count_parameters_number();
		num_net = num_net_;
		OpenNN::Vector<std::string> es(3,"HyperbolicTangent");
		perc->set_layers_activation_function(es);
		dset = new OpenNN::DataSet();
		vdset=std::vector<OpenNN::DataSet>(num_net,OpenNN::DataSet());
		num_vdset = std::vector<int>(num_net,0);

		// To Modify
		//connection = perc->get_layers_parameters();


	}

	//


	NNApproximator(int num_input_,int num_net_ = 1) {

		num_input = num_input_;
		hidden_neurons_number = num_input;
		//hidden_neurons_number = 1;
		outputs_number = 1;
		perc = new OpenNN::MultilayerPerceptron(num_input,
				hidden_neurons_number, outputs_number);
		num_training = perc->count_parameters_number();
		num_net = num_net_;

		//OpenNN::Vector<std::string> es(3,"Logistic");
		OpenNN::Vector<std::string> es(3,"HyperbolicTangent");
	    perc->set_layers_activation_function(es);
	    dset = new OpenNN::DataSet();
	    vdset=std::vector<OpenNN::DataSet>(num_net,OpenNN::DataSet());
	    num_vdset = std::vector<int>(num_net,0);

		// To Modify
		//connection = perc->get_layers_parameters();

	}

	// Constructor with any architecture

	NNApproximator(std::vector<int>& architecture, int num_net_ = 1){

		OpenNN::Vector<unsigned int> net;
		for(int i= 0; i < (int)architecture.size();i++){

			net.push_back(architecture[i]);
		}
		num_input = net[0];
		hidden_neurons_number = net[1];
		outputs_number = 1;
		num_net = num_net_;

		perc = new OpenNN::MultilayerPerceptron(net);
		num_training = perc->count_parameters_number();

		OpenNN::Vector<std::string> es(net.size(),"HyperbolicTangent");
		//OpenNN::Vector<std::string> es(net.size(),"Logistic");
		//es[es.size()-1]="Linear";
		perc->set_layers_activation_function(es);

		dset = new OpenNN::DataSet();
		vdset=std::vector<OpenNN::DataSet>(num_net,OpenNN::DataSet());
		num_vdset = std::vector<int>(num_net,0);

	}


	virtual double predict(VectorXd& input, VectorXd& weight){


		OpenNN::Vector<double> connection;
		connection.reserve(num_training);
		//std::cout<<num_training<<std::endl;

		//std::cerr<<"num t "<<num_training<<"num w "<<weight.rows()<<std::endl;

		for(int i = 0; i < num_training; i++){

			connection.push_back((double)weight(i));
		}


		perc->set_parameters(connection);

		OpenNN::Vector<double> in;
		for(int i=0; i < num_input; i++){

			in.push_back((double)input(i));
		}

		OpenNN::Vector<double> tmp = perc->calculate_outputs(in);

		return tmp[0];

	}

	virtual VectorXd predict2(VectorXd& input, VectorXd& weight) {


				//VectorXd rep=VectorXd::Zero(num_net);
				VectorXd rep=-2*VectorXd::Ones(num_net);
				VectorXd w=VectorXd::Zero(num_training);
				for(int i=0; i< num_net;i++){
				
				 if(num_vdset[i] <=0)
		                continue;

					for(int j= 0; j<num_training;j++){
						w(j)= weight(i*num_training+j);
					}

					rep(i)=predict(input,w);

				}

				return rep;
	}

	virtual void set_training(int num_instance_, std::vector<VectorXd >& d,  vi& Q_act_){

		//dset->set_instances_number(num_instance_);
		//dset->set_variables_number(num_input);
		Q_act = Q_act_;
		if(num_net==1){
			Q_act=vi(Q_act_.size(),0);
		}
		dset->set(num_instance_,num_input,outputs_number);



		for(int i=0; i < (int)Q_act.size(); i++){

			num_vdset[Q_act[i]]++;


		}

		for(int i=0; i < num_net;i++){
		
		/// This means that if their is no action of type i
		
		        if(num_vdset[i] >0)

			vdset[i].set(num_vdset[i],num_input,outputs_number);
		}

		OpenNN::Vector<double> d2(num_input);
		std::vector<int> num_tmp(num_net,0);

		for(int i=0;i<num_instance_;i++){

			for(int j=0; j<num_input;j++){

				d2[j] = d[i][j];
			}

			dset->set_training_input_instance(i,d2);

			vdset[Q_act[i]].set_training_input_instance(num_tmp[Q_act[i]],d2);
			num_tmp[Q_act[i]]++;


		}





	}


	virtual VectorXd fit(VectorXd& target){

		VectorXd rep=VectorXd::Zero(num_training*num_net);
		//VectorXd rep=-5000*VectorXd::Ones(num_training*num_net);
		OpenNN::Vector<double> tar(1);
		std::vector<int> num_tmp(num_net,0);

					for (int i = 0; i < (int) dset->get_instances_number(); i++) {

						tar[0] = target(i);

						vdset[Q_act[i]].set_target_instance(num_tmp[Q_act[i]], tar);
						num_tmp[Q_act[i]]++;
					}
		for(int k=0; k< num_net;k++){
		
		/// This means that if their is no action of type i
		
		        if(num_vdset[k] <=0)
		                continue;



			perc->initialize_parameters_uniform();

			OpenNN::NeuralNetwork *nn = new OpenNN::NeuralNetwork(*perc);

			OpenNN::PerformanceFunctional* perf =
					new OpenNN::PerformanceFunctional(nn, &vdset[k]);

			perf->set_regularization_term_flag(false);
			perf->set_constraints_term_flag(false);
			perf->set_objective_term_flag(false);

			OpenNN::GradientDescent *algo = new OpenNN::GradientDescent(perf);
			//OpenNN::QuasiNewtonMethod *algo = new OpenNN::QuasiNewtonMethod(perf);

			algo->set_maximum_time(900);
			algo->set_display_period(200);
			algo->set_performance_goal(0.000000000000000000000000000000002);
			algo->set_maximum_epochs_number(20);
			algo->set_display(false);

			//std::cout<<"Starting"<<std::endl;

			//algo->get_training_rate_algorithm_pointer()->set_training_rate_method(
				//	OpenNN::TrainingRateAlgorithm::Fixed);

			OpenNN::GradientDescent::GradientDescentResults* res =
					algo->perform_training();

			//OpenNN::QuasiNewtonMethod::QuasiNetwonMethodResults *res=algo->perform_training();

			//algo->perform_training();

			//std::cout << "final res " << res->final_evaluation << std::endl;
			//OpenNN::Vector<double> rep2 = res->final_parameters;
			OpenNN::Vector<double> rep2 = nn->arrange_parameters();
			//std::cerr<<"co"<<rep2.size();
			for (int i = 0; i < num_training; i++) {
				rep(k*num_training+i) = rep2[i];
				//std::cerr<<"rt";
				//rep(i) = 2;
			}


		}
		//std::cout<<"ans "<<rep<<std::endl;

		return rep;


	}

	/*virtual VectorXd fit(VectorXd& target){

		VectorXd rep=VectorXd::Zero(num_training);

		//OpenNN::DataSet* dset = new OpenNN::DataSet(data.get_rows_number(),num_input,outputs_number);

		//dset->set_data(data);

		OpenNN::Vector<double> tar(1);

		for(int i=0; i< (int)dset->get_instances_number();i++){

			tar[0] = target(i);

			dset->set_target_instance(i,tar);
		}
		perc->initialize_parameters_uniform();

		OpenNN::NeuralNetwork *nn=new OpenNN::NeuralNetwork(*perc);


		OpenNN::PerformanceFunctional* perf = new OpenNN::PerformanceFunctional(nn,dset);

		OpenNN::GradientDescent *algo = new OpenNN::GradientDescent(perf);

		algo->set_maximum_time(200);
		algo->set_display_period(200);
		algo->set_performance_goal(0.00002);
		algo->set_maximum_epochs_number(40);
		algo->set_display(false);

		//std::cout<<"Starting"<<std::endl;

		algo->get_training_rate_algorithm_pointer()->set_training_rate_method(OpenNN::TrainingRateAlgorithm::Fixed);

		OpenNN::GradientDescent::GradientDescentResults* res=algo->perform_training();

		//algo->perform_training();

		std::cout<<"final res "<<res->final_evaluation<<std::endl;
		//OpenNN::Vector<double> rep2 = res->final_parameters;
		OpenNN::Vector<double> rep2 = nn->arrange_parameters();
		//std::cerr<<"co"<<rep2.size();
		for(int i=0; i<num_training;i++){
			rep(i) = rep2[i];
			//std::cerr<<"rt";
			//rep(i) = 2;
		}
		//std::cout<<"pol "<<rep<<std::endl;
		return rep;


	}*/




	virtual int get_num_training(){

				return num_training*num_net;
			}
	virtual ~NNApproximator(){

	}
};

#endif /* NNAPPROXIMATOR_H_ */
