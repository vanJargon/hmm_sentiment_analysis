# HMM for Tweet Sentiment Analysis
Implementation of HMM and Structured Perceptron algorithm for POS Tagging of tweets. 

##### Python version: 2.7.9 #####

Project Overview:
=================
This project is divided into 5 main parts as follows:
- Part 1: Annotation of POS tags for tweets
- Part 2: Decoding of POS tags using emission probabilities
- Part 3: Decoding of POS tags using HMM (Viterbi)
- Part 4: Decoding of POS tags using HMM (Forward-Backward Algorithm)
- Part 5: Decoding of POS tags using Structured Perceptron (Collins, 2002)

Usage Instructions:
====================
Download the file and change into the `code` directory. 

To run the scripts for Parts 2-4 for each dataset,

Use `-d` to specify the dataset that should be run (ie. `EN`, `FR`, `CN` or `SG`)

For instance, to run `part2.py` over the `EN` dataset, run:
```
python part2.py -d EN
```

For Part 5,

Use `-k` to specify the minimum number of times a word needs to be observed in the training dataset to not be replaced with `'##UNK##'`

Use `-i` to specify the number of iterations over the training data in the Structured Perceptron algorithm

For instance, to run `part5.py` over the `EN` dataset with `k=3` and `i=4`, run:
```
python part5.py -d EN -k 3 -i 4
```

For all parts, the output file will be saved in the directory of the respective dataset (e.g. `/EN`.

##### Important: Please do not change the structure of the file directory as the relative path of the training set, validation set and output file is specified in the script. #####

Evaluation Results 
===================
### Part 2 ###
#### Test Results for EN: ####
Entity in gold data: 226
Entity in prediction: 1201

Correct Entity : 165
Entity  precision: 0.1374
Entity  recall: 0.7301
Entity  F: 0.2313

Correct Sentiment : 71
Sentiment  precision: 0.0591
Sentiment  recall: 0.3142
Sentiment  F: 0.0995

#### Test Results for FR: ####
Entity in gold data: 223
Entity in prediction: 1149

Correct Entity : 182
Entity  precision: 0.1584
Entity  recall: 0.8161
Entity  F: 0.2653

Correct Sentiment : 68
Sentiment  precision: 0.0592
Sentiment  recall: 0.3049
Sentiment  F: 0.0991

#### Test Results for CN: ####
Entity in gold data: 362
Entity in prediction: 3318

Correct Entity : 183
Entity  precision: 0.0552
Entity  recall: 0.5055
Entity  F: 0.0995

Correct Sentiment : 57
Sentiment  precision: 0.0172
Sentiment  recall: 0.1575
Sentiment  F: 0.0310

#### Test Results for SG: ####
Entity in gold data: 1382
Entity in prediction: 6599

Correct Entity : 794
Entity  precision: 0.1203
Entity  recall: 0.5745
Entity  F: 0.1990

Correct Sentiment : 315
Sentiment  precision: 0.0477
Sentiment  recall: 0.2279
Sentiment  F: 0.0789

### Part 3 ###
#### Test Results for EN: ####
Entity in gold data: 226
Entity in prediction: 162

Correct Entity : 104
Entity  precision: 0.6420
Entity  recall: 0.4602
Entity  F: 0.5361

Correct Sentiment : 64
Sentiment  precision: 0.3951
Sentiment  recall: 0.2832
Sentiment  F: 0.3299

#### Test Results for FR: ####
Entity in gold data: 223
Entity in prediction: 166

Correct Entity : 112
Entity  precision: 0.6747
Entity  recall: 0.5022
Entity  F: 0.5758

Correct Sentiment : 72
Sentiment  precision: 0.4337
Sentiment  recall: 0.3229
Sentiment  F: 0.3702

#### Test Results for CN: ####
Entity in gold data: 362
Entity in prediction: 158

Correct Entity : 64
Entity  precision: 0.4051
Entity  recall: 0.1768
Entity  F: 0.2462

Correct Sentiment : 47
Sentiment  precision: 0.2975
Sentiment  recall: 0.1298
Sentiment  F: 0.1808

#### Test Results for SG: ####
Entity in gold data: 1382
Entity in prediction: 723

Correct Entity : 386
Entity  precision: 0.5339
Entity  recall: 0.2793
Entity  F: 0.3667

Correct Sentiment : 244
Sentiment  precision: 0.3375
Sentiment  recall: 0.1766
Sentiment  F: 0.2318

### Part 4 ###
#### Test Results for EN: ####
Entity in gold data: 226
Entity in prediction: 175

Correct Entity : 108
Entity  precision: 0.6171
Entity  recall: 0.4779
Entity  F: 0.5387

Correct Sentiment : 69
Sentiment  precision: 0.3943
Sentiment  recall: 0.3053
Sentiment  F: 0.3441

#### Test Results for FR: ####
Entity in gold data: 223
Entity in prediction: 173

Correct Entity : 113
Entity  precision: 0.6532
Entity  recall: 0.5067
Entity  F: 0.5707

Correct Sentiment : 73
Sentiment  precision: 0.4220
Sentiment  recall: 0.3274
Sentiment  F: 0.3687

### Part 5 ###
#### Additional Modifications: ####
- Pre-processing: Conversion of observations to lowercase in learning and testing phase

#### Test Results for EN: ####
Entity in gold data: 226
Entity in prediction: 323

Correct Entity : 160
Entity  precision: 0.4954
Entity  recall: 0.7080
Entity  F: 0.5829

Correct Sentiment : 86
Sentiment  precision: 0.2663
Sentiment  recall: 0.3805
Sentiment  F: 0.3133

##### Parameters used for EN: numIters=12, k=1 #####


#### Test Results for FR: ####
Entity in gold data: 223
Entity in prediction: 280

Correct Entity : 177
Entity  precision: 0.6321
Entity  recall: 0.7937
Entity  F: 0.7038

Correct Sentiment : 103
Sentiment  precision: 0.3679
Sentiment  recall: 0.4619
Sentiment  F: 0.4095

##### Parameters used for FR: numIters=4, k=1 #####
