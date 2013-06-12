/*
 * STEnvironment.h
 *
 *  Created on: 11 Sep 2012
 *      Author: yedtoss
 */

#ifndef STENVIRONMENT_H_
#define STENVIRONMENT_H_

#include "AppEnvironment.h"

class STEnvironment: public AppEnvironment {
public:

	int num_t;
    std::vector<VectorXd> v_basis, kernel_v_basis;
    std::vector<std::vector<VectorXd> > q_basis, kernel_q_basis;

    std::vector<VectorXmp> v_basis_bis, kernel_v_basis_bis;
    std::vector<std::vector<VectorXmp> > q_basis_bis, kernel_q_basis_bis;
	std::vector<std::vector<std::vector<double> > > tr;
	std::vector<std::vector<std::vector<int> > > succ;
	bool useState;

	STEnvironment(int n_states_, int n_actions_, int nf_, int num_t_,
			std::vector<VectorXd>&v_basis_,
			std::vector<std::vector<VectorXd> >& q_basis_,
			std::vector<std::vector<std::vector<int> > >& succ_,
            std::vector<std::vector<std::vector<double> > >&tr_, bool use_kernel_=false, Demonstration *D_=NULL, double kernel_sig_ = 1,int max_points_ = INT_MAX, int features_type_ = 2, App type_ =
					App_NN, IRL_MODE mode = IRL_CPRB, bool useState_ = false,
					int num_hidden = 1, vi layerlist = vi(1,4)) {

		n_states = n_states_;
		n_actions = n_actions_;
		n_features = nf_;
		num_t = num_t_;
		v_basis = v_basis_;
		q_basis = q_basis_;
		succ = succ_;
		tr = tr_;
		type = type_;
		useState = useState_;
        use_kernel = use_kernel_;
        kernel_sig = kernel_sig_;
        features_type = features_type_;
        compute_demo(*D_,max_points_);
        compute_basis();

        if(use_kernel){
        compute_kernel_matrix();
        compute_kernel_basis();
        }

		if (type == App_Linear) {

//			if(useState){

//				app = new LinearApproximator(n_features);
//				qapp = new LinearApproximator(n_features);

//			}
//			else {

//				app = new LinearApproximator(n_features);
//				qapp = new LinearApproximator(n_features * n_actions);
//			}

            if(!use_kernel){

            if(features_type == 0)
            {
                app = new LinearApproximator(n_features);
                qapp = new LinearApproximator(n_features*n_actions);

            }

            else if (features_type == 1) {
                app = new LinearApproximator(n_features);
                qapp = new LinearApproximator(n_features);
            }

            else if(features_type == 2){
                app = new LinearApproximator(n_features);
                qapp = new LinearApproximator(n_features*n_actions);
            }
            }
            else{

                if(features_type == 0)
                {
                    app = new LinearApproximator(n_points);
                    qapp = new LinearApproximator(n_points*n_actions);

                }

                else if (features_type == 1) {
                    app = new LinearApproximator(n_points);
                    qapp = new LinearApproximator(n_points);
                }

                else if(features_type == 2){
                    app = new LinearApproximator(n_points);
                    qapp = new LinearApproximator(n_points*n_actions);
                }

            }

			MultipleNN = false;
		}

		else if (type == App_NN) {

			vi netw(2+num_hidden, n_features);
			netw[1] = n_features;
			//netw[1] = 2/2;

			for(int i = 0; i<num_hidden; i++){

				netw[i+1] = layerlist[i];
			}
			netw[netw.size()-1] = 2/2;
			//netw[3] = 1;
			//if(netw[1] > n_features){
			//netw[1] = 10;
			//}
			//netw[2] = 1;

			if (mode == IRL_CPRB) {


				if(useState){


					app = new FNNApproximator(netw, ACT_TANH);
					//app = new LinearApproximator(n_features);
					//netw[1] = 3;
					qapp = new FNNApproximator(netw, ACT_TANH);



				}

				else {

					app = new FNNApproximator(netw, ACT_TANH);
					//app = new LinearApproximator(n_features);
					//netw[1] = 3;
					qapp = new FNNApproximator(netw, ACT_TANH, n_actions);

				}


			}

			else if (mode == IRL_CRPB) {


				if(useState){



					//app = new NNApproximator(netw);
					app = new LinearApproximator(n_features);
					//netw[1] = 30;
					//qapp = new NNApproximator(netw);
					qapp = new FNNApproximator(netw);


				}

				else {


					//app = new NNApproximator(netw);
					app = new LinearApproximator(n_features);
					//netw[1] = 30;
					//qapp = new NNApproximator(netw, n_actions);
					qapp = new FNNApproximator(netw, ACT_TANH, n_actions);

				}

			}

			if(useState){

				MultipleNN = false;
			}

			else {
				MultipleNN = true;
			}



		}



		//app = new NNApproximator(n_features);
		vapp = app;
		//qapp = app;
		rapp = app;
		//MultipleNN = false;
		//type = App_Linear;

	}

    virtual void compute_kernel_basis()
    {
        kernel_v_basis_bis = std::vector<VectorXmp>(n_states);
        for(int s=0; s <n_states; s++)
        {
            //kernel_v_basis_bis[s] = (((-kernel_sig*((kernel_matrix_bis.colwise()-v_basis_bis[s]).colwise().norm())).array().exp().matrix()-kernel_mean.transpose())*kernel_components).transpose();
            /// TO BE FIXED
            tmp4 = (kernel_matrix_bis.colwise()-v_basis_bis[s]);
            tmp4.resize(1,tmp4.size());
            kernel_v_basis_bis[s] = ((tmp4 -  kernel_mean.transpose())*kernel_components).transpose();

        }

        kernel_q_basis_bis = std::vector<std::vector<VectorXmp> >(n_states,std::vector<VectorXmp>(n_actions));

        for(int s=0; s<n_states; s++)
        {
            for(int a = 0; a < n_actions; a++)
            {
                //
                if(features_type >2)
                {
                    std::cerr<<" There is only 3 type of features falling back to type 2"<<std::endl;
                    features_type = 2;

                }

                // One group of features for each action
               if(features_type == 0)
               {
                   kernel_q_basis_bis[s][a] = VectorXmp(n_points*n_actions);
                   kernel_q_basis_bis[s][a].segment(n_points*a,n_points) =  kernel_v_basis_bis[s];
               }

               // Here the features is the same as the state with the
               // greatest probability


               else if(features_type == 1)
               {
                   double best =0;
                   int idx =0;
                   best = -5;
                   idx = 0;

                   for(int s_tr=0; s_tr<(int)tr[s][a].size(); s_tr++){


                       if(tr[s][a][s_tr] > best){

                           best = tr[s][a][s_tr];
                           idx = s_tr;
                       }


                   }
                   /// What if idx is the terminal state ?

                   if(best >0){

                       kernel_q_basis_bis[s][a] = kernel_v_basis_bis[succ[s][a][idx]];
                   }
                   else {

                       kernel_q_basis_bis[s][a] = kernel_v_basis_bis[s];
                   }


               }

               // Here the features is the group of all the features of the
               // successors states after the couple (s,a) is taken

               else if(features_type == 2)
               {

                   double best = -5;
                   int idx = 0;
                   kernel_q_basis_bis[s][a] = VectorXmp(n_points*n_actions);

                   for(int s_tr=0; s_tr<(int)tr[s][a].size(); s_tr++){


                       if(tr[s][a][s_tr] > best){

                           best = tr[s][a][s_tr];
                           idx = s_tr;
                       }
                       kernel_q_basis_bis[s][a].segment(s_tr*n_points, n_points) = kernel_v_basis_bis[succ[s][a][s_tr]];


                   }
                   /// What if (s,a) has less than n_actions transitions state ?
                   for(int s_tr = tr[s][a].size(); s_tr<n_actions; s_tr++)
                   {
                       kernel_q_basis_bis[s][a].segment(s_tr*n_points, n_points) = kernel_v_basis_bis[s];

                   }



               }

            }
        }

        kernel_v_basis = std::vector<VectorXd>(kernel_v_basis_bis.size());
        for(int i=0; i<kernel_v_basis_bis.size();i++)
        {
            kernel_v_basis[i] = kernel_v_basis_bis[i].cast<double>();
        }

        kernel_q_basis = std::vector<std::vector<VectorXd> >(kernel_q_basis_bis.size());

        for(int i=0; i< kernel_q_basis_bis.size();i++)
        {
            for(int j=0; j< kernel_q_basis_bis[i].size();j++)
            {
                kernel_q_basis[i].push_back(kernel_q_basis_bis[i][j].cast<double>());
            }
        }






    }

	virtual void compute_basis() {

		std::vector<std::vector<double> > stat(n_features,
				std::vector<double>(3, 0));

		for (int s = 0; s < n_states; s++) {

			for (int i = 0; i < n_features; i++) {

				stat[i][0] += v_basis[s][i];

				if (s == 0 || v_basis[s][i] < stat[i][1]) {
					stat[i][1] = v_basis[s][i];

				}

				if (s == 0 || v_basis[s][i] > stat[i][2]) {

					stat[i][2] = v_basis[s][i];
				}
			}

		}

		for (int s = 0; s < n_states; s++) {

			for (int i = 0; i < n_features; i++) {


				/*if(stat[i][2] == stat[i][1]){

					if(stat[i][2] != 0){
						v_basis[s][i] = v_basis[s][i]/stat[i][2];
					}
					else {

						v_basis[s][i] = 0.5;
					}
					continue;
				}


				v_basis[s][i] = (v_basis[s][i] - stat[i][0] / (double) n_states)
						/ (stat[i][2] - stat[i][1]);

				std::cout<<v_basis[s][i]<<"\t";*/
			}
		}
        if(!use_kernel){

        if (features_type == 1) {

			q_basis = std::vector<std::vector<VectorXd> >(n_states,
					std::vector<VectorXd>(n_actions,
							VectorXd::Zero(n_features)));

			double best = 0;
			int idx =0;

			for (int s = 0; s < n_states; s++) {

				for (int a = 0; a < n_actions; a++) {

					best = -5;
					idx = 0;

					for(int s_tr=0; s_tr<(int)tr[s][a].size(); s_tr++){


						if(tr[s][a][s_tr] > best){

							best = tr[s][a][s_tr];
							idx = s_tr;
						}


					}
					/// What if idx is the terminal state ?

					if(best >0){

						q_basis[s][a] = v_basis[succ[s][a][idx]];
					}
					else {

						q_basis[s][a] = v_basis[s];
					}

				}
			}

		}

		else {

			if (MultipleNN) {

				q_basis = std::vector<std::vector<VectorXd> >(n_states,
						std::vector<VectorXd>(n_actions,
								VectorXd::Zero(n_features)));

				for (int s = 0; s < n_states; s++) {

					for (int a = 0; a < n_actions; a++) {

						q_basis[s][a] = v_basis[s];

					}
				}

			}

            if(features_type == 0){

				q_basis = std::vector<std::vector<VectorXd> >(n_states,
						std::vector<VectorXd>(n_actions,
								VectorXd::Zero(n_features * n_actions)));

				for (int s = 0; s < n_states; s++) {

					for (int a = 0; a < n_actions; a++) {

                        q_basis[s][a].segment(a*n_features, n_features) = v_basis[s];

                        //for (int i = 0; i < n_features; i++) {

                            //q_basis[s][a](i + a * n_features) = v_basis[s](i);
                        //}
					}
				}

			}

            if(features_type == 2){
                double best = -5;
                int idx = 0;
                q_basis = std::vector<std::vector<VectorXd> >(n_states,
                        std::vector<VectorXd>(n_actions));

                for(int s=0; s<n_states; s++)
                {
                    for(int a=0; a<n_actions; a++)
                    {

                        q_basis[s][a] = VectorXd(n_features*n_actions);

                        for(int s_tr=0; s_tr<(int)tr[s][a].size(); s_tr++){


                            if(tr[s][a][s_tr] > best){

                                best = tr[s][a][s_tr];
                                idx = s_tr;
                            }
                            q_basis[s][a].segment(s_tr*n_features, n_features) = v_basis[succ[s][a][s_tr]];


                        }
                        /// What if (s,a) has less than n_actions transitions state ?
                        for(int s_tr = tr[s][a].size(); s_tr<n_actions; s_tr++)
                        {
                            q_basis[s][a].segment(s_tr*n_features, n_features) = v_basis[s];

                        }


                    }
                }



            }

		}
    }

		v_basis_bis = std::vector<VectorXmp>(v_basis.size());
        for(int i=0; i<(int)v_basis.size();i++)
        {
            v_basis_bis[i] = v_basis[i].cast<mpreal>();
        }

        q_basis_bis = std::vector<std::vector<VectorXmp> >(q_basis.size());

        for(int i=0; i< (int)q_basis.size();i++)
        {
            for(int j=0; j< (int)q_basis[i].size();j++)
            {
                q_basis_bis[i].push_back(q_basis[i][j].cast<mpreal>());
            }
        }

        // If kernel is activated let's precompute the features
//        if(use_kernel)
//        {
//            compute_kernel_basis();
//        }
	}

	virtual VectorXd& get_basis(int i_ep, int i_ns, DEMO_MODE demo_st, int s,
			int a = -1, int s_next = -1, bool q_features_available_ = false) {

		if (s_next != -1 && a != -1) {

            if(use_kernel)
            {
                return kernel_v_basis[s_next];
            }
            else
            {
                return v_basis[s_next];
            }

			//return q_basis[s][a];
		}

		else if (s_next == -1 && a != -1) {

			//if (MultipleNN) {

				//return v_basis[s];
			//} else {

				//return q_basis[s][a];
			//}
            if(use_kernel)
            {
                return kernel_q_basis[s][a];
            }
            else
            {
                return q_basis[s][a];
            }


		}
        if(use_kernel)
        {
            return kernel_v_basis[s];
        }

		return v_basis[s];
	}


	virtual VectorXmp& get_basis(int s,int a = -1, int s_next = -1) {

        if (s_next != -1 && a != -1) {

            if(use_kernel)
            {
                return kernel_v_basis_bis[s_next];
            }
            else
            {
                return v_basis_bis[s_next];
            }

            //return q_basis[s][a];
        }

        else if (s_next == -1 && a != -1) {

            //if (MultipleNN) {

                //return v_basis[s];
            //} else {

                //return q_basis[s][a];
            //}
            if(use_kernel)
            {
                return kernel_q_basis_bis[s][a];
            }
            else
            {
                return q_basis_bis[s][a];
            }


        }
        if(use_kernel)
        {
            return kernel_v_basis_bis[s];
        }

        return v_basis_bis[s];
	}

	virtual std::string Name() {

		return "CSEnvironment";
	}
	virtual ~STEnvironment() {

	}
};

#endif /* STENVIRONMENT_H_ */
