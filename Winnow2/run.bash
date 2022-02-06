#!/bin/bash

srcpath="./../data"
dir=$1 #input the directory name

#provide the names of train and test files
pos_trn_file=$2
neg_trn_file=$3
pos_tst_file=$4
neg_tst_file=$5

#a bit of cleaning
cat $srcpath/$dir/raw/$pos_trn_file | sed "s/>  //g" > train.pos;
cat $srcpath/$dir/raw/$neg_trn_file | sed "s/>  //g" > train.neg;
cat $srcpath/$dir/raw/$pos_tst_file | sed "s/>  //g" > test.pos;
cat $srcpath/$dir/raw/$neg_tst_file | sed "s/>  //g" > test.neg;

#prepare the data for Winnow2
#matlab -nodisplay -nojvm -batch 'preparedata'

#python winnow2.py
