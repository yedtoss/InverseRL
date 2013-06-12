/*
 * LinearApproximator.h
 *
 *  Created on: 7 Sep 2012
 *      Author: yedtoss
 */

#ifndef LINEARAPPROXIMATOR_H_
#define LINEARAPPROXIMATOR_H_



#include "Approximator.h"

class LinearApproximator : public Approximator {
public:
	LinearApproximator(int num_input_){
		num_training = num_input_;
	}

	virtual double predict(VectorXd& input, VectorXd& weight){

		return input.dot(weight);

	}

	virtual VectorXd gradient(VectorXd& input, VectorXd& weight){
			return input;
	}
	virtual ~LinearApproximator(){

	}
};

#endif /* LINEARAPPROXIMATOR_H_ */
