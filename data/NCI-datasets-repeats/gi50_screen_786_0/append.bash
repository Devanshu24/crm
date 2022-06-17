#!/bin/bash

# A script to append BotGNN predicted labels with every instance of the NCI datasets
# BotGNN is constructed with Graph-SAGE. Refer our paper on:
# "Inclusion of Symbolic-Domain Knowledge Using Mode-Directed Inverse Entailment"

datadir='.'
splitdir='./../NCI-datasets/splitinfo'
labeldir='./../NCI-datasets/BotGNN_preds/gi50_screen_786_0'
resultdir='./withBotGNNlabels'

for i in {4..4}
do
	cat $labeldir/train_preds | sed 's/1,1/pos,ppos/g' | sed 's/1,0/pos,pneg/g' | sed 's/0,1/neg,ppos/g' | sed 's/0,0/neg,pneg/g' > trnlabels;
	cat $labeldir/test_preds | sed 's/1,1/pos,ppos/g' | sed 's/1,0/pos,pneg/g' | sed 's/0,1/neg,ppos/g' | sed 's/0,0/neg,pneg/g' > tstlabels;

	grep pos, trnlabels > train.p
	grep neg, trnlabels > train.n
	grep pos, tstlabels > test.p
	grep neg, tstlabels > test.n

	dataset="crm_r$i"
	datapath="$datadir/$dataset/2_0_0.5_10"
	paste -d, $datapath/rand_crm_d3c3v6r$i\_train_features_pos train.p > trn
	paste -d, $datapath/rand_crm_d3c3v6r$i\_train_features_neg train.n >> trn
	paste -d, $datapath/rand_crm_d3c3v6r$i\_test_features_pos test.p > tst
	paste -d, $datapath/rand_crm_d3c3v6r$i\_test_features_neg test.n >> tst

	cat trn | grep ",ppos" | sed "s/,pos,ppos//g" | sed "s/,neg,ppos//g" > train_pos
	cat trn | grep ",pneg" | sed "s/,pos,pneg//g" | sed "s/,neg,pneg//g" > train_neg
	cat tst | grep ",ppos" | sed "s/,pos,ppos//g" | sed "s/,neg,ppos//g" > test_pos
	cat tst | grep ",pneg" | sed "s/,pos,pneg//g" | sed "s/,neg,pneg//g" > test_neg

	mkdir -p $resultdir/$dataset
	mv train_* $resultdir/$dataset/.
	mv test_* $resultdir/$dataset/.

	rm trnlabels tstlabels *.p *.n trn tst
done
