#!/bin/bash

#cat ./data/NCI-datasets-repeats/datasets | while read dataset;
#do
    dataset="gi50_screen_A498"
    echo $dataset
    datadir="./data/NCI-datasets-repeats/$dataset"
    #datadir="./data/786_0_depth5/crm_repeats"

    for i in {5..5}
    do
	    echo "$i"
	    fileprefix="rand_crm_d3c3v6r1"
	    ln -s $datadir/crm_r$i/2_0_0.5_10/$fileprefix.pl crm_structure.pl
	    ln -s $datadir/crm_r$i/2_0_0.5_10/$fileprefix\_train_features_pos train_pos
	    ln -s $datadir/crm_r$i/2_0_0.5_10/$fileprefix\_train_features_neg train_neg
	    ln -s $datadir/crm_r$i/2_0_0.5_10/$fileprefix\_test_features_pos test_pos
	    ln -s $datadir/crm_r$i/2_0_0.5_10/$fileprefix\_test_features_neg test_neg

	    #python main.py -f ./repeat-config -o $datadir/crm_r$i/2_0_0.5_10/res_max_expl -n 10 -e -v #predictions with explanations
	    #python main.py -f ./repeat-config -o $datadir/crm_r$i/2_0_0.5_10/res_max_expl -n 10 -p -v #just the predictions
	    #python main.py -f ./repeat-config  -o $datadir/crm_r$i/2_0_0.5_10/res_pred -n 0 -s $datadir/crm_r$i/2_0_0.5_10/res_max_expl_model_1.pt -p -v
	    python main.py -f ./repeat-config-tr  -o $datadir/crm_r$i/2_0_0.5_10/res_pred_tr -n 0 -s $datadir/crm_r$i/2_0_0.5_10/res_max_expl_model_5.pt -p -v

	    #wc -l crm_structure.pl train_pos train_neg test_pos test_neg
	    rm crm_structure.pl train_pos train_neg test_pos test_neg
    done
#done


#datadir="./data/786_0_repeats"
#
#for i in {4..4}
#do
#	echo "$i"
#	ln -s $datadir/crm_r$i/2_0_0.5_10/rand_crm_d3c3v6r$i.pl crm_structure.pl
#	ln -s $datadir/withBotGNNlabels/crm_r$i/train_pos
#	ln -s $datadir/withBotGNNlabels/crm_r$i/train_neg
#	ln -s $datadir/withBotGNNlabels/crm_r$i/test_pos
#	ln -s $datadir/withBotGNNlabels/crm_r$i/test_neg
#
#	python main.py -f ./repeat-config -o $datadir/withBotGNNlabels/crm_r$i/res_max_expl -n 10 -e -v -t
#
#	#wc -l crm_structure.pl train_pos train_neg test_pos test_neg
#	rm crm_structure.pl train_pos train_neg test_pos test_neg
#done
