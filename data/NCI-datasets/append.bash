#!/bin/bash

# A script to append BotGNN predicted labels with every instance of the NCI datasets
# BotGNN is constructed with Graph-SAGE. Refer our paper on:
# "Inclusion of Symbolic-Domain Knowledge Using Mode-Directed Inverse Entailment"

datadir='.'
splitdir='./splitinfo'
labeldir='./BotGNN_preds'
resultdir='./withBotGNNlabels'

for dataset in `cat datasets | tail -1`
do
	cat $labeldir/$dataset/train_preds | sed 's/1,1/pos,ppos/g' | sed 's/1,0/pos,pneg/g' | sed 's/0,1/neg,ppos/g' | sed 's/0,0/neg,pneg/g' > trnlabels;
	cat $labeldir/$dataset/test_preds | sed 's/1,1/pos,ppos/g' | sed 's/1,0/pos,pneg/g' | sed 's/0,1/neg,ppos/g' | sed 's/0,0/neg,pneg/g' > tstlabels;

	grep pos, trnlabels > train.p
	grep neg, trnlabels > train.n
	grep pos, tstlabels > test.p
	grep neg, tstlabels > test.n

	datapath="$datadir/$dataset/crm/2_0_0.5_10"
	paste -d, $datapath/rand_crm_d3c3v6r1_train_features_pos train.p > trn
	paste -d, $datapath/rand_crm_d3c3v6r1_train_features_neg train.n >> trn
	paste -d, $datapath/rand_crm_d3c3v6r1_test_features_pos test.p > tst
	paste -d, $datapath/rand_crm_d3c3v6r1_test_features_neg test.n >> tst

	cat trn | grep ",ppos" | sed "s/,pos,ppos//g" | sed "s/,neg,ppos//g" > train_pos
	cat trn | grep ",pneg" | sed "s/,pos,pneg//g" | sed "s/,neg,pneg//g" > train_neg
	cat tst | grep ",ppos" | sed "s/,pos,ppos//g" | sed "s/,neg,ppos//g" > test_pos
	cat tst | grep ",pneg" | sed "s/,pos,pneg//g" | sed "s/,neg,pneg//g" > test_neg

	mkdir -p $resultdir/$dataset
	mv train_* $resultdir/$dataset/.
	mv test_* $resultdir/$dataset/.

	rm trnlabels tstlabels *.p *.n trn tst
done
