#ifndef UTILS_H
#define UTILS_H

#include "MyTypes.h"
#include "AppEnvironment.h"
#include <fstream>

vdd get_q(VectorXd &opt_policy, AppEnvironment& env) {

    //q_true.resize(env->get_n_states(),
        //	std::vector<double>(env->get_n_actions()));

    vdd q_true = vdd(env.get_n_states(),
            std::vector<double>(env.get_n_actions(),0));

    for (int i = 0; i < env.get_n_states(); i++) {

        for (int j = 0; j < env.get_n_actions(); j++) {

            q_true[i][j] = env.get_q(opt_policy, -1, -1, DEMO_NONE, i, j);
        }
    }

    return q_true;

}

vdd get_r(VectorXd &opt_rew, AppEnvironment& env) {

    //r_true.resize(env->get_n_states(),
        //	std::vector<double>(env->get_n_actions()));

    vdd r_true = vdd(env.get_n_states(),
            std::vector<double>(env.get_n_actions(),0));

    for (int i = 0; i < env.get_n_states(); i++) {

        for (int j = 0; j < env.get_n_actions(); j++) {

            r_true[i][j] = env.get_r(opt_rew, -1, -1, DEMO_NONE, i, -1);
        }
    }

    return r_true;

}

vd get_v(VectorXd &opt_policy, AppEnvironment& env) {

    //v_true.resize(env->get_n_states());
    vd v_true = vd(env.get_n_states());

    vdd q_true = get_q(opt_policy, env);


    for (int i = 0; i < env.get_n_states(); i++) {

        v_true[i] = *std::max_element(q_true[i].begin(), q_true[i].end());

    }

    return v_true;
}

vi get_policy_greedy(VectorXd &opt_policy, AppEnvironment& env) {

    //p_true.resize(env->get_n_states());

    vi p_true = vi(env.get_n_states());

    for (int i = 0; i < env.get_n_states(); i++) {

        p_true[i] = env.policy_act(opt_policy, i, true);
    }

    return p_true;

}

vdd  get_policy(VectorXd &opt_policy, AppEnvironment& env){

    vdd p_prob = vdd(env.get_n_states(),std::vector<double>(env.get_n_actions(),0));





    vdd q_true = get_q(opt_policy,env);

    double sum = 0;

    for(int s=0; s< env.get_n_states(); s++){

        for(int a=0; a<env.get_n_actions();a++){

            sum+=std::exp(q_true[s][a]);
        }

        for(int a=0; a<env.get_n_actions();a++){

            p_prob[s][a]=std::exp(q_true[s][a])/sum;
        }



    }

    return p_prob;
}

// Scaling features

void scale(std::vector<VectorXd>&features, double lower=0, double upper =1)
{
    assert(features.size() > 1);
    std::cout<<"INF = "<<INF<<std::endl;
    std::vector<double> features_min(features[0].rows(),INF);
    std::vector<double> features_max(features[0].rows(),-INF);

    for(int i=0; i< (int)features.size();i++)
    {
        for(int j=0; j < (int)features[0].rows(); j++)
        {
            if(features[i][j] < features_min[j])
            {
                features_min[j] = features[i][j];
            }

            if(features[i][j] > features_max[j])
            {
                features_max[j] = features[i][j];
            }
        }
    }

    for(int i=0; i< (int)features.size();i++)
    {
        for(int j=0; j < (int)features[0].rows(); j++)
        {
            if(features_max[j] == features_min[j]){
                continue;
            }

            if(features[i][j] == features_min[j]){
                features[i][j] = lower;
            }

            else if(features[i][j] == features_max[j]){
                features[i][j] = upper;
            }

            else{
                features[i][j] = lower+ (upper-lower)*(features[i][j]-features_min[j])/(features_max[j]-features_min[j]);




            }
        }
    }

}

void save_demo(Demonstration& demo, std::string s_file, int format =0, int time_step = 7, int pad_state = 200, int pad_action = 0){

    std::ofstream demout((s_file).c_str());
    if(format == 0){

        // Saving the demonstration file for LRP/LPR
        for (int i = 0; i < demo.size(); i++) {

            for(int t=0; t<demo[i].size(); t++){

                demout<<demo[i][t].s<<"\t"<<demo[i][t].a<<"\t"<<demo[i][t].s_next<<"\t";

                if(t == demo[i].size()-1){
                    demout<<";"<<"\t";
                }
                else{
                    demout<<","<<"\t";
                }
            }
        }

    }

    if(format == 1){

        // Saving the demonstration file for FIRL/MaxEnt
        // Note that we make sure to always have 7 time steps

        for (int i = 0; i < demo.size(); i++) {

            for(int t=0; t<demo[i].size(); t++){

                demout<<demo[i][t].s<<"\t"<<demo[i][t].a<<"\t";

            }

            // If we don't have 7 time steps we complete with
            // the terminal state

            for(int t = demo[i].size(); t<time_step; t++){

                demout<<pad_state<<"\t"<<pad_action<<"\t";
            }

        }
    }

    demout.close();
}

Demonstration load_demo(std::string s_file){

    Demonstration demo;
    std::ifstream demo_in((s_file).c_str());

    int s,a,s_next;
    std::string sep;
    while(demo_in>>s>>a>>s_next>>sep){

        demo.Observe(s,a,s_next);
        if(sep==";"){
            demo.NewEpisode();
        }
    }

    demo_in.close();

    return demo;



}



#endif // UTILS_H
