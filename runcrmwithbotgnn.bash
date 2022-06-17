#!/bin/bash

datadir="./data/NCI-datasets"
#
#for dataset in `cat $datadir/datasets | tail -1`
#do
        dataset="gi50_screen_DMS_114"
        echo "Running CRM for $dataset"

        ln -s $datadir/$dataset/crm/2_0_0.5_10/rand_crm_d3c3v6r1.pl crm_structure.pl
        ln -s $datadir/withBotGNNlabels/$dataset/train_pos
        ln -s $datadir/withBotGNNlabels/$dataset/train_neg
        ln -s $datadir/withBotGNNlabels/$dataset/test_pos
        ln -s $datadir/withBotGNNlabels/$dataset/test_neg

        #python main.py -f ./NCI-config -o $datadir/withBotGNNlabels/$dataset/res_max_expl_gs -n 10 -t
        python main.py -f ./NCI-config -o $datadir/withBotGNNlabels/$dataset/newtest_res -n 0 -s $datadir/withBotGNNlabels/$dataset/res_max_expl_gs_model_0.pt -v

        rm *.pl train_pos train_neg test_pos test_neg
#done
