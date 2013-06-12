#ifndef LRP_H_INCLUDED
#define LRP_H_INCLUDED
//#include "stdafx.h"
//#include "optimization.h"
#include "AppEnvironment.h"
#include "MyTypes.h"
#include "LSPI.h"


//#include "mpfr.h"
//#include "mpreal.h"

#include <cassert>
#include <time.h>
#include <lbfgs.h>




using mpfr::exp;
using mpfr::log;
int digits = 1024;


int counter =0;
VectorXd ans_final;
VectorXmp ans_final2;
const mpreal INF2 = INF;
mpreal bestv = INF2;
mpreal beta = 1.0/5.0;
mpreal beta_inv = 1.0/beta;

double xl_bound = -1;
double xh_bound = 1;
int n_bfgs_corr = 3;
int max_iterations = 200;
double f_epsilon = 1e-9;
double gr_epsilon = 1e-9;
double step_min_step = 1e-20;
double step_max_step = 1e+20;
int linesearch_type = 0;
int max_linesearch = 40;
VectorXmp w_r_bis, grad_bis;

class NLP2
{

    /** default constructor */

public:

    Demonstration *D;
    AppEnvironment *env;
    Demo_C demo_comp ;

    MatrixXmp b1,A_I,C;
    int k_p,k_r;
    // tmp keeps C*w_r.dot(phi), tmp_exp keeps it's exponential
    VectorXmp tmp,tmp_exp, tmp_den, tmp_den_square, tmp_grad;

    //MatrixXmp tmp_grad;

    mpreal grad_sum, sum, value, value_obj, v1, v;
    mpreal EPS = 1e-10;
    ALG algo;


    NLP2(){
    }


    NLP2(Demonstration &D_, AppEnvironment& env_, ALG algo_=ALG_LPR)
    {
        mpfr::mpreal::set_default_prec(mpfr::digits2bits(digits));



        D = &D_;
        env = &env_;
        algo = algo_;
        k_p = env->get_q_num_training();
        k_r = env->rapp->get_num_training();
        std::cerr<<"INF2inity is "<<INF2<<std::endl;
        if(2000 < INF2)
        {

            std::cerr<<"Success"<<std::endl;
        }

        if(-2000 > -INF2)
        {

            std::cerr<<"Success"<<std::endl;
        }

        // This size is a trick to avoid problem
        // In fact it is possible that all states do not have the same
        // number of actions. We multiply by 2 and add 10 to the default
        // number return when no states is given to get_n_actions

        // We should change the behaviour of get_n_actions to return the
        // maximum number of actions
        tmp= VectorXmp::Zero(env->get_n_actions()*2+10);
        tmp_exp = VectorXmp::Zero(env->get_n_actions()*2+10);

        tmp_grad = MatrixXmp::Zero(k_r,1);

        precompute();



    }


    void precompute()
    {

        // PreComputing matrix A of LSPI
        /// We are using  only policy evaluation
        /// so we can precompute A and b1 = \phi_q \phi_r'


        int a_t = 0, s,a,s_next,nep,ns,i;
        if(algo == ALG_LRP){
        b1 = MatrixXmp::Zero(k_p,k_r);
        double reg = 2;
        double gamma2 = 1;

        MatrixXmp A= reg*MatrixXmp::Identity(k_p,k_p);

        //C = MatrixXmp::Zero(k_p,k_r);



        std::cerr<<"Starting computation of A and b"<<std::endl;
        // Parcourir la demonstration
        for(nep =0; nep<(*D).size(); nep++)
        {

            for(ns = 0; ns < (*D)[nep].size(); ns++)
            {


                s = (*D)[nep][ns].s;
                a = (*D)[nep][ns].a;
                s_next = (*D)[nep][ns].s_next;

                if (s_next <= -1)
                {
                    continue;
                }

                VectorXmp& phi = env->get_q_basis(s,
                                                 a);


                a_t = 0;
                //std::cerr<<"phi("<<s<<","<<a<<")"<<phi.transpose()<<std::endl;

                if (ns + 1 != (*D)[nep].size())
                {

                    a_t = (*D)[nep][ns + 1].a;
                }

                VectorXmp& phi_next = env->get_q_basis(s_next, a_t);

                VectorXmp& phi_r = env->get_v_basis(s, a, s_next);

                A.noalias() += phi * ((phi - gamma2 * phi_next).transpose());

                b1.noalias() += phi*phi_r.transpose();
            }



        }


        /// Let's find inverse of A
        std::cerr<<"Computing inverse of A"<<std::endl;

        Eigen::FullPivLU<MatrixXmp> lu2;
        lu2.compute(A);
        A_I = lu2.solve(MatrixXmp::Identity(k_p,k_p));
        std::cerr<<"Computing C"<<std::endl;

        C.noalias() = A_I * b1;
        }
        else if(algo == ALG_LPR){

            k_r = k_p;
            std::cerr<<"Computing C"<<std::endl;
            C = MatrixXmp::Identity(k_p,k_r);

        }
        else {
            std::cerr<<"Error you should choose between LPR or LRP"<<std::endl;
            exit(-1);
        }
        //std::cerr<<" C = "<<C<<std::endl;


        std::cerr<<"Computing Compress Demonstration"<<std::endl;

        demo_comp = compress(*D);

        /// Precomputing the C.col(i)*phi
        std::cerr<<"Computing C.col(i)*phi coefficient"<<std::endl;

        demo_comp.grad = VectorXmp::Zero(k_r);

        for(nep=0; nep<demo_comp.size(); nep++)
        {
            s = demo_comp[nep].s;

            // Computing a vector containing the number of times each actions
            // occured

            demo_comp[nep].num_vec = VectorXmp(demo_comp[nep].size());

            for(ns = 0; ns < demo_comp[nep].size(); ns++)
            {
                demo_comp[nep].num_vec[ns] = demo_comp[nep][ns].num;
            }

            demo_comp[nep].coeff_dot = MatrixXmp(env->get_n_actions(s),k_r);
            for(a =0; a < env->get_n_actions(s);a++)
            {
                //demo_comp[nep].coeff_dot[a] = VectorXmp::Zero(k_r);
                //a = demo_comp[nep][ns].a;
                VectorXmp& phi = env->get_q_basis(s, a);
                demo_comp[nep].coeff_dot.row(a).noalias() = beta_inv*(C.transpose())*phi;

            }

            //std::cerr<<"coeff dot "<<demo_comp[nep].coeff_dot<<std::endl;


            for(ns = 0; ns < demo_comp[nep].size(); ns++)
            {
                a = demo_comp[nep][ns].a;

                for(i = 0; i<k_r;i++)
                {
                    demo_comp.grad[i] += demo_comp[nep][ns].num*(demo_comp[nep].coeff_dot(a,i));
                }

            }
        }
        //std::cerr<<"demo_comp grad"<<demo_comp.grad.transpose()<<std::endl;
        std::cerr<<"End of all precomputation"<<std::endl;

    }


    /// Given a reward Lets compute sum_D Q(s_i,a_i)-V(s_i) and it's gradient

     void objectives(VectorXmp& w_r, double& value2)
    {
        value = 0;
        int s, a, n_act;

        for(int nep =0; nep<demo_comp.size(); nep++)
        {

            s = demo_comp[nep].s;

            n_act = env->get_n_actions(s);

            if(n_act <=0)
            {
                continue;
            }


            tmp.noalias() = ((demo_comp[nep].coeff_dot*w_r));
            sum =  beta*log((tmp/beta).array().exp().sum());

            for(int ns=0; ns < demo_comp[nep].size(); ns++)
            {
                a = demo_comp[nep][ns].a;


                value +=  demo_comp[nep][ns].num*(tmp[a]);
            }

            value -= sum*demo_comp[nep].tot_act;

        }

        value = -value;
        //std::cout<<"   val "<<value;
        //std::cout<<"      w_r "<<w_r;

        value2 = value.toDouble();

        if(value < bestv){

            ans_final2 = w_r;
            bestv = value;
        }




    }

    void obj_grad(VectorXmp& w_r, VectorXmp& grad)
    {

        int s, a, n_act;

        //grad = VectorXmp::Zero(k_r);
        grad.setZero();
        grad += demo_comp.grad;

        for(int nep =0; nep<demo_comp.size(); nep++)
        {

            s = demo_comp[nep].s;

            n_act = env->get_n_actions(s);

            if(n_act <=0)
            {
                continue;
            }


            tmp_exp.noalias() = ((demo_comp[nep].coeff_dot*w_r)/beta).array().exp().matrix();


            grad.noalias() -=(demo_comp[nep].coeff_dot.transpose()*tmp_exp)*(demo_comp[nep].tot_act/tmp_exp.sum());

        }

        grad = -grad;
        //std::cerr<<"     grad "<<grad.transpose()<<std::endl;
    }

    void obj_grad_value(VectorXmp& w_r, VectorXmp& grad, double& value2)
    {

        value_obj = 0;
        int s, a, n_act;
        grad.setZero();
        grad.noalias() -= demo_comp.grad;

        for(int nep =0; nep<demo_comp.size(); nep++)
        {

            s = demo_comp[nep].s;

            n_act = env->get_n_actions(s);

            if(n_act <=0)
            {
                continue;
            }


            tmp.noalias() = ((demo_comp[nep].coeff_dot*w_r));
            tmp_exp.noalias() = (tmp).array().exp().matrix();
            sum = log(tmp_exp.sum());
            //sum =  log((tmp).array().exp().sum());

            for(int ns=0; ns < demo_comp[nep].size(); ns++)
            {
                a = demo_comp[nep][ns].a;


                value_obj -=  demo_comp[nep][ns].num*(tmp[a]);
            }

            value_obj += sum*demo_comp[nep].tot_act;

            //tmp_exp.noalias() = ((demo_comp[nep].coeff_dot*w_r)).array().exp().matrix();


            grad.noalias() +=(demo_comp[nep].coeff_dot.transpose()*tmp_exp)*(demo_comp[nep].tot_act/tmp_exp.sum());

        }

        //value = -value;
        //std::cout<<"   val "<<value;
        //std::cout<<"      w_r "<<w_r;

        value2 = value_obj.toDouble();

        if(value_obj < bestv){

            ans_final2 = w_r;
            bestv = value_obj;
        }


    }


    /*void obj_grad_value(VectorXmp& w_r, VectorXmp& grad, double& value2)
    {

        value_obj = 0;
        int s, a, n_act;
        grad.setZero();

        for(int nep =0; nep<demo_comp.size(); nep++)
        {

            s = demo_comp[nep].s;

            n_act = env->get_n_actions(s);

            if(n_act <=0)
            {
                continue;
            }


            tmp.noalias() = ((demo_comp[nep].coeff_dot*w_r));
            tmp_exp.noalias() = (tmp/beta).array().exp().matrix();
            sum =  beta*log(tmp_exp.sum());

            tmp_den_square.noalias() =(tmp.array().square()+EPS).matrix();
            tmp_den.noalias() = (tmp_den_square).array().sqrt().matrix();

            //tmp_grad.noalias() -=(demo_comp[nep].coeff_dot.transpose()*tmp_exp)*(demo_comp[nep].tot_act/tmp_exp.sum());

            tmp_grad.noalias()=(demo_comp[nep].coeff_dot.transpose()*tmp_exp)/(tmp_exp.sum());
            v = 0;

            for(int ns=0; ns < demo_comp[nep].size(); ns++)
            {
                a = demo_comp[nep][ns].a;


                //value +=  demo_comp[nep][ns].num*(tmp[a]);
                //den = (sqrt(pow(tmp[a],2)+EPS));
                v1 = sum-tmp[a];


                v+= demo_comp[nep][ns].num*(v1)/tmp_den[a];
                grad +=  demo_comp[nep][ns].num*(((tmp_grad-demo_comp[nep].coeff_dot.row(a).transpose()) * (tmp_den[a]) - (v1)*tmp[a]*demo_comp[nep].coeff_dot.row(a).transpose()/tmp_den[a])/(tmp_den_square[a]));
            }
            value_obj += v;

            //value -= sum*demo_comp[nep].tot_act;








        }



        //value = -value;
        //std::cout<<"   val "<<value;
        //std::cout<<"      w_r "<<w_r;

        value2 = value_obj.toDouble();

        if(value_obj < bestv){

            ans_final2 = w_r;
            bestv = value_obj;
        }


    }*/

    void obj_hess(VectorXmp& w_r, MatrixXmp& hess)
    {
        int s, a, n_act;

        hess = MatrixXmp::Zero(k_r,k_r);

        for(int nep =0; nep<demo_comp.size(); nep++)
        {

            s = demo_comp[nep].s;

            n_act = env->get_n_actions(s);

            if(n_act <=0)
            {
                continue;
            }

            mpreal sum_square=0;
            sum =0;
            //tmp= VectorXmp::Zero(n_act);
            //tmp_exp= VectorXmp::Zero(n_act);
            //double value = 0;


            for (a = 0; a < n_act; a++)
            {
                VectorXmp& phi = env->get_q_basis(s, a);
                tmp[a] = (C*w_r).dot(phi);
                tmp_exp[a] = exp(tmp[a]/beta);

                sum += tmp_exp[a];

            }

            sum_square = sum*sum;

            //mpreal tot = 0;

            //for(int ns=0; ns < demo_comp[nep].size(); ns++)
            //{
              //  tot+= demo_comp[nep][ns].num;

            //}

            MatrixXmp coef = MatrixXmp::Zero(k_r,n_act);

            for(int i=0; i<k_r;i++){
                for(a=0; a<n_act; a++){

                    //VectorXmp& phi = env->get_q_basis(s, a);
                    //coef(i,a) = beta_inv*(C.col(i)).dot(phi);
                    coef(i,a) = beta_inv*demo_comp[nep].coeff_dot(a,i);


                }

            }

            //MatrixXd tmp_hess = MatrixXd::Zero(k_r,k_r);

            mpreal v1,v2,v3;

            for(int i=0; i<k_r;i++){


                for(int j=0; j<=i;j++){

                        v1 =0;


                        for(a=0; a<n_act; a++){
                            //VectorXmp& phi = env->get_q_basis(s, a);

                            v1 += coef(i,a)*coef(j,a)*tmp_exp[a];

                        }

                        //tmp_hess(i,j) = v*sum;

                        v2 =0;
                        v3 =0;


                        for(a=0; a<n_act; a++){
                            //VectorXmp& phi = env->get_q_basis(s, a);

                            v2 += coef(i,a)*tmp_exp[a];
                            v3 += coef(j,a)*tmp_exp[a];

                        }

                        hess(i,j) += demo_comp[nep].tot_act*( v2*v3- v1*sum)/sum_square;



                }
            }



        }

        hess = -hess;


    }





    /** default destructor */
    virtual ~NLP2()
    {

    }




};

void init_var(int &k_r)
{
    w_r_bis = VectorXmp::Zero(k_r);
    grad_bis = VectorXmp::Zero(k_r);
}

void take_options()
{
    std::ifstream in2("opt");
    std::string str;
    while(in2>>str){

        if(str=="beta_inv"){

            in2>>beta_inv;
            beta = 1.0/beta_inv;
        }

        else if(str=="digits"){

            in2>>digits;
        }

        else if(str=="xl_bound"){

            in2>>xl_bound;
        }

        else if(str=="xh_bound"){

            in2>>xh_bound;
        }

        else if(str=="n_bfgs_corr")
        {
            in2>>n_bfgs_corr;
        }

        else if(str=="max_iterations")
        {
            in2>>max_iterations;
        }

        else if(str=="f_epsilon")
        {
            in2 >> f_epsilon;
        }

        else if(str=="gr_epsilon")
        {
            in2 >> gr_epsilon;
        }

        else if(str=="step_min_step")
        {
            in2 >> step_min_step;
        }

        else if(str=="step_max_step")
        {
            in2 >> step_max_step;
        }

        else if(str=="linesearch_type")
        {
            in2 >> linesearch_type;
        }
        else if(str=="max_linesearch")
        {
            in2 >> max_linesearch;
        }
    }

}

NLP2 nlp2;


static lbfgsfloatval_t evaluate_lbfgs(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
    )
{


    //VectorXmp w_r = VectorXmp::Zero(nlp2.k_r);
    lbfgsfloatval_t fx = 0.0;
    double func = 0;
    for(int i=0; i<nlp2.k_r; i++)
    {
        w_r_bis[i] = x[i];
    }
    //nlp2.objectives(w_r_bis,func);



    //nlp2.obj_grad(w_r_bis,grad_bis);

    nlp2.obj_grad_value( w_r_bis, grad_bis,func);

    for(int i=0; i< nlp2.k_r; i++)
    {
        g[i] = grad_bis[i].toDouble();
    }

    fx = func;
    return fx;
}

static int progress_lbfgs(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
    )
{
    printf("Iteration %d: ", k);
    printf("  fx = %f", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

double computeLRP(Demonstration &D_, AppEnvironment& env_,VectorXd& ans, ALG algo=ALG_LPR)
{
    take_options();
    int time_start = clock();
    nlp2=NLP2(D_,env_,algo);

    int N = nlp2.k_r;

    init_var(N);

    int  ret = 0;
    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(N);
    lbfgs_parameter_t param;

    if (x == NULL) {
        printf("ERROR: Failed to allocate a memory block for variables.\n");
        return -1;
    }

    /* Initialize the variables. */
    for (int i = 0;i < N;i += 1) {
        x[i] = 1.0/N;
    }

    /* Initialize the parameters for the L-BFGS optimization. */
    lbfgs_parameter_init(&param);
    param.max_iterations = max_iterations;
    param.m = n_bfgs_corr;
    param.epsilon = gr_epsilon;
    param.delta = f_epsilon;
    param.min_step = step_min_step;
    param.max_step = step_max_step;
    param.linesearch = linesearch_type;
    param.max_linesearch = max_linesearch;


    /*param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;*/

    /*
        Start the L-BFGS optimization; this will invoke the callback functions
        evaluate() and progress() when necessary.
     */

    ret = lbfgs(N, x, &fx, evaluate_lbfgs, progress_lbfgs, NULL, &param);
    int time_end = clock();

    std::cout<<param.epsilon<<" " <<param.delta<<" "<<param.max_iterations<<std::endl;

    /* Report the result. */
    printf("L-BFGS optimization terminated with status code = %d\n", ret);
    printf("  fx = %f, ", fx);
    for(int i=0; i<N; i++)
    {
        printf("  x[%d] = %f,  ",i,x[i]);
    }
    printf("\n");

    std::cout<<std::endl<<std::endl<<"Found solution :"<<std::endl;



    VectorXmp ans_fin = nlp2.C*ans_final2;
    ans_final = VectorXd::Zero(nlp2.k_p);

    for(int i=0; i<nlp2.k_p; i++){

        ans_final[i] = ans_fin[i].toDouble();
        std::cout<<ans_fin[i]<<std::endl;
    }

    std::cout << std::endl << std::endl << "Final best objective =" << bestv<<std::endl;

    ans = ans_final;
    // testing to disable
    //ans[nlp2.k_p/2-1] = -23000;
    //ans[nlp2.k_p-1] = 23000;

    // Quick test to check if doing LSPI after convergence improve results
    /*LSPI lspi(2,1e-90,3000);
    ApproximateMDP mdp;
    mdp.R = ans_final2.cast<double>();
    lspi.compute(ans,env_,D_,mdp,false);*/



    lbfgs_free(x);

    return (time_end-time_start)/(double)CLOCKS_PER_SEC;

}

#endif // LRP_H_INCLUDED
