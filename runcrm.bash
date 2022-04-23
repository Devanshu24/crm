#!/bin/bash

#datadir="./data/NCI-datasets"
#
#for dataset in `cat $datadir/datasets | tail -1`
#do
#	echo "Running CRM for $dataset"
#
#	ln -s $datadir/$dataset/crm/2_0_0.5_10/rand_crm_d3c3v6r1.pl
#	ln -s $datadir/withBotGNNlabels/$dataset/train_pos
#	ln -s $datadir/withBotGNNlabels/$dataset/train_neg
#	ln -s $datadir/withBotGNNlabels/$dataset/test_pos
#	ln -s $datadir/withBotGNNlabels/$dataset/test_neg
#
#	#python main.py -f ./NCI-config -o $datadir/withBotGNNlabels/$dataset/res_max_expl_gs -n 10 -t
#	python main.py -f ./NCI-config -o $datadir/withBotGNNlabels/$dataset/res_max_expl_gs1 -n 0 -s $datadir/withBotGNNlabels/$dataset/res_max_expl_gs_model_4.pt -e -v
#
#	rm *.pl train_pos train_neg test_pos test_neg
#done

datadir="./data/NCI-datasets"

for dataset in `cat $datadir/datasets | head -9`
do
	echo "Running CRM for $dataset"
	path="$datadir/$dataset/crm/2_0_0.5_10"

	ln -s $path/rand_crm_d3c3v6r1.pl crm_structure.pl
	ln -s $path/rand_crm_d3c3v6r1_train_features_pos train_pos
	ln -s $path/rand_crm_d3c3v6r1_train_features_neg train_neg
	ln -s $path/rand_crm_d3c3v6r1_test_features_pos test_pos
	ln -s $path/rand_crm_d3c3v6r1_test_features_neg test_neg

	python main.py -f ./NCI-config -o $path/res_max_expl_gs -n 10 -t

	rm crm_structure.pl train_pos train_neg test_pos test_neg
done

#datadir="./data/786_0_repeats"
#
#for i in {1..1} #{1..3}
#do
#	echo "$i"
#	ln -s $datadir/crm_r$i/2_0_0.5_10/rand_crm_d3c3v6r$i.pl crm_structure.pl
#	ln -s $datadir/crm_r$i/2_0_0.5_10/rand_crm_d3c3v6r$i\_train_features_pos train_pos
#	ln -s $datadir/crm_r$i/2_0_0.5_10/rand_crm_d3c3v6r$i\_train_features_neg train_neg
#	ln -s $datadir/crm_r$i/2_0_0.5_10/rand_crm_d3c3v6r$i\_test_features_pos test_pos
#	ln -s $datadir/crm_r$i/2_0_0.5_10/rand_crm_d3c3v6r$i\_test_features_neg test_neg
#
#	python main.py -f ./repeat-config -o $datadir/crm_r$i/2_0_0.5_10/res_max_expl -n 10 -e -v
#
#	#wc -l crm_structure.pl train_pos train_neg test_pos test_neg
#	rm crm_structure.pl train_pos train_neg test_pos test_neg
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
