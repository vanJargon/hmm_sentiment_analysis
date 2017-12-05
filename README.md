# Machine Learning Project
HMM for tweet sentiment analysis


Part 2
------
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

Part 3
------
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

Part 4
------
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

Part 5
------
Improvement of Viterbi algorithm (Part 3).

### Modifications made: ###
- Conversion of observations to lowercase in learning and testing phase
- Implementation of Structured Perceptron algorithm instead of HMM

#### Test Results for EN: ####
Entity in gold data: 226
Entity in prediction: 306

Correct Entity : 142
Entity  precision: 0.4641
Entity  recall: 0.6283
Entity  F: 0.5338

Correct Sentiment : 71
Sentiment  precision: 0.2320
Sentiment  recall: 0.3142
Sentiment  F: 0.2669

#### Test Results for FR: ####
Entity in gold data: 223
Entity in prediction: 93

Correct Entity : 70
Entity  precision: 0.7527
Entity  recall: 0.3139
Entity  F: 0.4430

Correct Sentiment : 45
Sentiment  precision: 0.4839
Sentiment  recall: 0.2018
Sentiment  F: 0.2848

#### Test Results for CN: ####
Entity in gold data: 362
Entity in prediction: 581

Correct Entity : 143
Entity  precision: 0.2461
Entity  recall: 0.3950
Entity  F: 0.3033

Correct Sentiment : 96
Sentiment  precision: 0.1652
Sentiment  recall: 0.2652
Sentiment  F: 0.2036

