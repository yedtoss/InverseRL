/*
 * GreedyGQ.h
 *
 *  Created on: 23 Oct 2012
 *      Author: yedtoss
 */
 
 
 #ifndef OffPac_H_
#define OffPac_H_


#include "TDLearning.h"


class OffPac : public TDLearning {
public:

        double lambda2,alpha2,beta2,alphat;
        
        VectorXd e,w,v,et;
        double rho2;
        int astar;


        OffPac(double gamma_ =1, double lambda_ = 0){

		init(gamma_,lambda_);

	}

	void init(double& gamma_,double& lambda_) {

		//threshold = threshold_;
		//num_iterations = num_iterations_;
		gamma2 = gamma_;
		lambda2 = lambda_;
		alpha2 = 0.00005;
		beta2 = 0.00025;
		fast = false;
		alphat = 0.002;
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
		
		et = VectorXd::Zero(n_features);
		
		v = VectorXd::Zero(n_features);
	
	
	}
	
	
	virtual void update(VectorXd& theta, AppEnvironment& env, Demonstration& D, ApproximateMDP& mdp,
					VectorXd* pi = NULL, VectorXd* b = NULL){
					
					//std::cout<<" b ";
					
		VectorXd& phi = env.get_q_basis(nep, ns, DEMO_CURRENT, s, a);
		VectorXd& phi_next = env.get_q_basis(nep, ns, DEMO_ACT_NEXT, s_next,env.policy_act(
				theta, s_next, true, nep, ns, DEMO_ACT_NEXT) );
	        delta = env.get_r(mdp.R, nep, ns, DEMO_NEXT, s, a, s_next)+ gamma2*v.dot(phi_next)-v.dot(phi);
	        
	        astar = env.policy_act(theta, s, true, nep, ns, DEMO_ACT_NEXT);
	        
	        rho2 =0;
	        
	        if(astar == a ){
	        
	                rho2 = 1.0/(env.get_n_actions(s));
	                assert(!std::isnan(rho2));
			assert(!std::isinf(rho2));
	        
	        
	        }
	        
	        
	        e = rho2*(phi + gamma2*lambda2*e);
	        
	        
	        for(int i=0; i<e.rows();i++){
	        assert(!std::isnan(e(i)));
	       assert(!std::isinf(e(i)));
	       }
	        
	        v = v + alpha2*(delta*e-gamma2*(1-lambda2)*phi.dot(w)*phi_next);
	        
	        
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
	       
	       
	       if(astar == a)
	       
	       
	       et = rho2*(gamma2*lambda2*et);
	       
	       else
	       
	       et = rho2*(500*phi-500*env.get_q_basis(nep, ns, DEMO_CURRENT, s, astar)+gamma2*lambda2*et);
	       
	       std::cout<<et<<std::endl;
	       
	       
	       theta = theta + alphat * delta * et;
	        
	        
					
					
	}
};





#endif /* OffPac_H_ */
