/*
 * Random2.h
 *
 *  Created on: 9 Sep 2012
 *      Author: yedtoss
 */

#ifndef RANDOM2_H_
#define RANDOM2_H_

//#include "Utility.h"

#include "Matrix.h"
//#include "Environment.h"
#include "MersenneTwister.h"
#include "Distribution.h"
#include "NormalDistribution.h"
//#include "Demonstration.h"
#include "Dirichlet.h"

//#include "dlib/optimization.h"
#include <iostream>

class Random2 {
public:
	DirichletDistribution *dirichlet;
	MersenneTwisterRNG *mersenne_twister;
	RandomNumberGenerator* rng;
	BernoulliDistribution *bernoulli;
	NormalDistribution *gauss;
	Vector y;
	VectorXd rgen;
	Random2(int nf_ = 2, double p_ =2) {
		srand48(time(NULL));
		srand(time(NULL));

		setRandomSeed(time(NULL));

		mersenne_twister = new MersenneTwisterRNG();
		rng = (RandomNumberGenerator*) mersenne_twister;
		rng->manualSeed(time(NULL));
		dirichlet = new DirichletDistribution(nf_,p_);
		y.Resize(nf_);
		rgen = VectorXd::Zero(nf_);
		bernoulli = new BernoulliDistribution(0.5);
		gauss = new NormalDistribution();

	}
	;

	double uniform(double l = 0, double u = 1) {
		return rng->uniform(l, u);
	}

	/*Matrix generateDirichlet() {
		Matrix rep;
		return rep;
	}*/

	VectorXd& generateBernoulli(){

		double val;

		for(int i =0; i<rgen.rows(); i++){
			val = bernoulli->generate();
			if(val == 0)
				val = -1;
			rgen(i) = val;

		}
		return rgen;
	}

	VectorXd& generateGaussian(int nf_,double mean_ = 0, double var_ = 50){

		gauss->setMean(mean_);
		gauss->setVariance(var_);
		for(int i =0; i< rgen.rows(); i++){

			rgen(i) = gauss->generate();
		}

		return rgen;





	}

	VectorXd& generateDirichlet() {

		dirichlet->generate(y);
		int sign;
		for(int i = 0; i < rgen.rows(); i++){
			sign = rng->discrete_uniform(2);
			rgen(i) = y(i);
			if(sign == 0){
				rgen(i) = - rgen(i);
			}
		}
		return rgen;

	}

	VectorXd& generateUnity(int nrow, bool isMinus = true, double l = 0,
			double u = 1) {
		//VectorXd rep = VectorXd::Zero(nrow);
		for (int r = 0; r < nrow; r++) {
			rgen(r) = rng->uniform(l, u);
		}
		double sum = rgen.sum();
		if(sum>0){
			//rep = rep / sum;
		}


		if (isMinus) {
			int sign;

			for (int r = 0; r < nrow; r++) {

				sign = rng->discrete_uniform(2);
				if (sign == 0) {
					rgen(r) = -rgen(r);
				}

			}

		}

		return rgen;

	}
	/*Matrix generateUnity(int nrow, int ncol, bool isMinus = true, double l = 0,
			double u = 1) {
		double sum = 0;
		Matrix rep(nrow, ncol);
		for (int r = 0; r < nrow; r++) {
			for (int c = 0; c < ncol; c++) {
				rep(r, c) = rng->uniform(l, u);
				sum += rep(r, c);
			}
		}

		for (int r = 0; r < nrow; r++) {
			for (int c = 0; c < ncol; c++) {
				//rep(r,c) = rep(r, c) / sum;
			}
		}
		if (isMinus) {
			int sign;

			for (int r = 0; r < nrow; r++) {
				for (int c = 0; c < ncol; c++) {
					sign = rng->discrete_uniform(2);
					if (sign == 0) {
						rep(r, c) = -rep(r, c);
					}
				}
			}

		}

		return rep;

	}*/

	virtual ~Random2() {
	}
	;
protected:
private:
};

#endif /* RANDOM2_H_ */
