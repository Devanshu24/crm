#!/bin/bash

rm analysis.out
cat analysis | while read line;
do
	echo $line
	grep $line ./raw/rand_crm_1npd3nrs_test_features_neg >> analysis.out;
done
