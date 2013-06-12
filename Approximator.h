/*
 * Approximator.h
 *
 *  Created on: 7 Sep 2012
 *      Author: yedtoss
 */

#ifndef APPROXIMATOR_H_
#define APPROXIMATOR_H_

#include <Eigen/Dense>
#include <iostream>

class Approximator {
public:
	int num_training;
	//Approximator(){

	//}

	virtual double predict(VectorXd& input, VectorXd& weight){
		return 0;
	}

	virtual VectorXd gradient(VectorXd& input, VectorXd& weight){
		return VectorXd::Zero(2);
	}

	virtual int get_num_training(){

		return num_training;
	}
	virtual VectorXd predict2(VectorXd& input, VectorXd& weight) {
			return VectorXd::Zero(2);
	}
	virtual void set_training(int num_instance_, std::vector<VectorXd >& d, vi& Q_act){

	}
	virtual VectorXd fit(VectorXd& target){

		return VectorXd::Zero(2);

	}

	virtual double get_min(){
		return -1;
	}
	virtual double get_max(){
		return 1;
	}
	virtual ~Approximator(){

	}
};

#endif /* APPROXIMATOR_H_ */
