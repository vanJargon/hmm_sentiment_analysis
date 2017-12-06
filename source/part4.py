#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Code for 01.112 Machine Learning Project (Part 4)

Done by:
Vanessa Tan (1001827)
Shruthi Shangar (1001630)
"""
import argparse

def estimateEmission(filePath, k=3):
    tags = {}  # count of the number of times a particular tag has appeared
    observations = {}  # count of the number of times a particular observation has appeared (irrespective of tag)
    l_Observations = {}  # count of labelled observations
    estimates = {}

    # Process the data
    for line in open(filePath, 'r'):
        segmentedLine = line.rstrip()
        if segmentedLine:  # if its not just an empty string
            segmentedLine = segmentedLine.rsplit(' ', 1)

            observation = segmentedLine[0]  # X
            tag = segmentedLine[1]  # Y

            if observation not in observations:  # if this observation has never been seen before
                observations[observation] = 1
            else:  # if this observation has been seen before
                observations[observation] += 1

            if tag not in tags:  # if this tag has never been seen before
                tags[tag] = 1
                l_Observations[tag] = {observation: 1}

            else:  # if this tag has been seen before
                tags[tag] += 1
                if observation not in l_Observations[tag]:
                    l_Observations[tag][observation] = 1
                else:
                    l_Observations[tag][observation] += 1

    # Compute the MLE for observations which appeared more than k times and for ##UNK##
    for tag in l_Observations:
        estimates[tag] = {}
        l_Observations[tag]['##UNK##'] = 0
        for observation in list(l_Observations[tag]):  # loop over all keys in l_Observations
            if observation == '##UNK##': continue
            if observation not in observations:  # if this observation has been found to appear less than k times before
                l_Observations[tag]['##UNK##'] += l_Observations[tag].pop(observation)
            elif observations[observation] < k:  # if first meet an observation that appear less than k times
                l_Observations[tag]['##UNK##'] += l_Observations[tag].pop(observation)
                del observations[observation]
            else:  # compute the MLE for that emission
                estimates[tag][observation] = float(l_Observations[tag][observation]) / tags[tag]
        estimates[tag]['##UNK##'] = float(l_Observations[tag]['##UNK##']) / tags[tag]

    return list(observations), estimates

def estimateTransition(filePath):
    tags = {}  # count the number of times a particular y(i-1) tag has appeared
    t_Tags = {}  # count the number of times a particular transition from y(i) to y(i-1) has appeared
    estimates = {}

    previousState = ''
    currentState = '##START##'

    # Process the data
    for line in open(filePath, 'r'):
        previousState = currentState if (currentState != '##STOP##') else '##START##'  # y(i-1)
        segmentedLine = line.rstrip()

        if segmentedLine:  # if its not just an empty string
            segmentedLine = segmentedLine.rsplit(' ', 1)
            currentState = segmentedLine[1]  # y(i)
        else:  # if an empty string is seen
            if previousState == '##START##': break  # training data always terminates with 2 empty lines
            currentState = '##STOP##'  # y(i)

        if previousState not in tags:  # if tag y(i-1) has never been seen before
            tags[previousState] = 1
            t_Tags[previousState] = {currentState: 1}
        else:
            tags[previousState] += 1
            if currentState not in t_Tags[previousState]:
                t_Tags[previousState][currentState] = 1
            else:
                t_Tags[previousState][currentState] += 1

    # Compute the MLE for transitions
    for tag in t_Tags:
        estimates[tag] = {}
        for transition in t_Tags[tag]:
            estimates[tag][transition] = float(t_Tags[tag][transition]) / tags[tag]

    return estimates

def sentimentAnalysis(inputPath, m_training, emissionEstimates, transitionEstimates, outputPath):
    """ splits test file into separate observation sequences and feeds them into Viterbi algorithm """
    f = open(outputPath, 'w')

    observationSequence = []
    for line in open(inputPath, 'r'):
        observation = line.rstrip()
        if observation:
            observationSequence.append(observation)
        else:
            predictionSequence = maxMarginal(observationSequence, m_training, emissionEstimates, transitionEstimates)
            for i in range(len(observationSequence)):
                if predictionSequence:
                    f.write('%s %s\n' % (observationSequence[i], predictionSequence[i]))
                else:  # for those rare cases where the final probability is all 0
                    f.write('%s O\n' % observationSequence[i])

            f.write('\n')
            observationSequence = []

    print 'Finished writing to file %s' % (outputPath)
    return f.close()

def maxMarginal(observationSequence, m_training, emissionEstimates, transitionEstimates):
    """ Max-Marginal Decoding with Forward-backward """
    tags = list(emissionEstimates)

    def getAlpha():
        """ Finds the values of alpha"""
        values = [{tag:0.0 for tag in tags} for o in observationSequence]

        # Base case
        for c_tag in tags:
            if c_tag not in transitionEstimates['##START##']: continue  # update tags which can be transitioned from ##START##

            values[0][c_tag] = transitionEstimates['##START##'][c_tag]

        # Recursive case
        for j in range(1, len(observationSequence)):
            for c_tag in tags:
                for p_tag in tags:
                    if c_tag not in transitionEstimates[p_tag]: continue  # only add to sum tags which can transition to c_tag

                    if observationSequence[j-1] in m_training:  # if this is not ##UNK##
                        if observationSequence[j-1] not in emissionEstimates[p_tag]: continue # skip if this emission cannot be found

                        # add if this emission can be found!
                        values[j][c_tag] += values[j - 1][p_tag] * transitionEstimates[p_tag][c_tag] * emissionEstimates[p_tag][observationSequence[j - 1]]
                    else:  # if this is ##UNK##
                        values[j][c_tag] += values[j - 1][p_tag] * transitionEstimates[p_tag][c_tag] * emissionEstimates[p_tag]['##UNK##']

        return values

    def getBeta():
        """ Finds the values of beta """
        values = [{tag: 0.0 for tag in tags} for o in observationSequence]

        # Base case
        for p_tag in tags:
            if '##STOP##' not in transitionEstimates[p_tag]: continue  # update tags which can transition to ##STOP##

            if observationSequence[-1] in m_training:  # if this word is not ##UNK##
                if observationSequence[-1] not in emissionEstimates[p_tag]: continue  # skip if this emission cannot be found

                # add if this emission can be found!
                values[-1][p_tag] = transitionEstimates[p_tag]['##STOP##'] * emissionEstimates[p_tag][observationSequence[-1]]
            else:  # if this is ##UNK##
                values[-1][p_tag] = transitionEstimates[p_tag]['##STOP##'] * emissionEstimates[p_tag]['##UNK##']

        # Recursive case
        for j in reversed(range(len(observationSequence)-1)):
            for p_tag in tags:
                for c_tag in tags:
                    if c_tag not in transitionEstimates[p_tag]: continue  # only add to sum c_tags which can be transitioned from this p_tag

                    if observationSequence[j] in m_training:  # if this word is not ##UNK##
                        if observationSequence[j] not in emissionEstimates[p_tag]: continue  # move to next p_tag if this emission cannot be found

                        # include emission probability if this emission can be found!
                        values[j][p_tag] += values[j + 1][c_tag] * transitionEstimates[p_tag][c_tag] * emissionEstimates[p_tag][observationSequence[j]]
                    else:  # if this is ##UNK##
                        values[j][p_tag] += values[j + 1][c_tag] * transitionEstimates[p_tag][c_tag] * emissionEstimates[p_tag]['##UNK##']

        return values

    # Initialization
    alphas = getAlpha()
    betas = getBeta()

    ## DEBUG
    # print '\n[DEBUG UTIL] All values displayed should be equal!'
    # for j in range(len(observationSequence)):
    #     print observationSequence[j]
    #     sum = 0.0
    #     for tag in tags:
    #         sum += alphas[j][tag] * betas[j][tag]
    #     print sum
    # print '\n'

    # After initializing, just use alphas and betas to make the prediction
    prediction = []
    for i in range(len(observationSequence)):
        result = [0.0, '']
        for tag in tags:
            score = alphas[i][tag] * betas[i][tag]
            if score > result[0]:
                result = [score, tag]
        prediction.append(result[1])
    return prediction

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, dest='dataset', help='Dataset to run script over', required=True)

    args = parser.parse_args()

    trainFilePath = '../%s/train' % (args.dataset)
    inputTestFilePath = '../%s/dev.in' % (args.dataset)
    outputTestFilePath = '../%s/dev.p4.out' % (args.dataset)

    transitionEstimates = estimateTransition(trainFilePath)
    m_training, emissionEstimates = estimateEmission(trainFilePath)
    sentimentAnalysis(inputTestFilePath, m_training, emissionEstimates, transitionEstimates, outputTestFilePath)

