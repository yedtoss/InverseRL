/*
 * GreedyGQ.h
 *
 *  Created on: 23 Oct 2012
 *      Author: yedtoss
 */
 
 
 #ifndef GREEDYGQ_H_
#define GREEDYGQ_H_


#include "TDLearning.h"


class GreedyGQ : public TDLearning {
public:

        double lambda2,alpha2,beta2;
        
        VectorXd e,w;
        double rho2;
        int astar;


        GreedyGQ(double gamma_ =1, double lambda_ = 0.2){

		init(gamma_,lambda_);

	}

	void init(double& gamma_,double& lambda_) {

		//threshold = threshold_;
		//num_iterations = num_iterations_;
		gamma2 = gamma_;
		lambda2 = lambda_;
		alpha2 = 0.0005;
		beta2 = 0.0025;
		fast = false;
	}
	
	
	void initialize(VectorXd& theta){
	
	        if(theta.rows()>0){
			//pi = VectorXd::Zero(n_features);
			theta = VectorXd::Random(n_features);
			theta = VectorXd::Zero(n_features);
		}
		else {
			
			theta = VectorXd::Random(n_features);
			theta = VectorXd::Zero(n_features);
		}
		
		w = VectorXd::Zero(n_features);
		
		e = VectorXd::Zero(n_features);
	
	
	}
	
	
	virtual void update(VectorXd& theta, AppEnvironment& env, Demonstration& D, ApproximateMDP& mdp,
					VectorXd* pi = NULL, VectorXd* b = NULL){
					
					//std::cout<<" b ";
					
		VectorXd& phi = env.get_q_basis(nep, ns, DEMO_CURRENT, s, a);
		VectorXd& phi_next = env.get_q_basis(nep, ns, DEMO_ACT_NEXT, s_next,env.policy_act(
				theta, s_next, true, nep, ns, DEMO_ACT_NEXT) );
	        delta = env.get_r(mdp.R, nep, ns, DEMO_NEXT, s, a, s_next)+ gamma2*theta.dot(phi_next)-theta.dot(phi);
	        
	        astar = env.policy_act(theta, s, true, nep, ns, DEMO_ACT_NEXT);
	        
	        rho2 =0;
	        
	        if(astar == a ){
	        
	                rho2 = 1.0/(env.get_n_actions(s));
	                assert(!std::isnan(rho2));
			assert(!std::isinf(rho2));
	        
	        
	        }
	        
	        
	        e = phi + rho2*gamma2*lambda2*e;
	        
	        
	        for(int i=0; i<e.rows();i++){
	        assert(!std::isnan(e(i)));
	       assert(!std::isinf(e(i)));
	       }
	        
	        theta = theta + alpha2*(delta*e-gamma2*(1-lambda2)*phi.dot(w)*phi_next);
	        
	        
	         for(int i=0; i<w.rows();i++){
	        assert(!std::isnan(theta(i)));
	       assert(!std::isinf(theta(i)));
	       }
	       
	      // std::cout<<theta<<std::endl<<w<<std::endl<<delta<<std::endl<<beta2<<std::endl;
	       //std::cout<<beta2<<" "<<gamma2<<" "<<lambda2<<delta<<std::endl;
	       
	        
	        
	        w = w + beta2*(delta*e-phi.dot(w)*phi);
	        
	        for(int i=0; i<w.rows();i++){
	        assert(!std::isnan(w(i)));
	       assert(!std::isinf(w(i)));
	       }
	        
	        
					
					
	}
};





#endif /* GREEDYGQ_H_ */
