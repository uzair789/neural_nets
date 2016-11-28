-To run the DBM training, run the script 'DBM.m'

-'DBM.m' will create a folder on your desktop and save all the images and plots. Make sure you change the path of this folder
according to your own settings.

-I am using a variable called 'plot_count'in DBM.m which is only to keep track of the runs of my experiments and save different images for 
each run of the experiments. If youdont change the value of this variable then the plots will keep saving with the same name and hence you
will lose the results from the previous run.

-I am also using a variable called 'live_plotting' only to visualize the error values while training. Set this variable to 'no' to not visualize
error while training as this will make your training plocess slow.

-'to sample from the DBM, run the script 'sampling.m'. This script will initialize 100 chains and do gibbs sampling for 1000 steps. 


-- To run the single layer RBM with Persistant Divergence, run 'RBM_PD.m' and use the script 'sampling_rbm.m' to sample from this model.

A description for all the scripts

CD.m 			---> 	Contrastive Divergence
cross_entropy_loss.m 	---> 	Computes the cross entropy loss
DBM.m 			---> 	runs the DBM training
loadData1.m 		---> 	loads the data into the train, val and test structures
PD.m			---> 	Persistence Divergence
RBM_PD.m		---> 	Runs the one layer RBM training with Persistance Divergence
sampling.m 		---> 	for sampling from the DBM model
sampling_rbm.m 		---> 	for sampling from the one layer rbm model
shuffle.m 		---> 	shuffles the data
sigmoid.m 		---> 	computes the sigmoid activation
view.m 			---> 	To view the weights and the data
visualize.m 		---> 	returns the matrix of the weights which can be viewed using imshow. 'view.m' calls 'visualize.m' internally.

DBM_Images 		---> Contains all the plots from the experiments
