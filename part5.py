#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:21:14 2017

@author: 1001827
"""
dataset = 'EN'
# dataset = 'FR'
# dataset = 'CN'
# dataset = 'SG'

trainFilePath = '../%s/train' % (dataset)
inputTestFilePath = '../%s/dev.in' % (dataset)
outputTestFilePath = '../%s/dev.p5.out' % (dataset)

# Same as part 2
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

            observation = segmentedLine[0].lower()  # X
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

    # print tags
    # print observations
    # print l_Observations
    # print estimates
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
            observationSequence.append(observation.lower())
        else:
            predictionSequence = viterbi(observationSequence, m_training, emissionEstimates, transitionEstimates)
            for i in range(len(observationSequence)):
                if predictionSequence:
                    f.write('%s %s\n' % (observationSequence[i], predictionSequence[i]))
                else:  # for those rare cases where the final probability is all 0
                    f.write('%s O\n' % observationSequence[i])

            f.write('\n')
            observationSequence = []

    print 'Finished writing to file %s' % (outputPath)
    return f.close()

def viterbi(observationSequence, m_training, emissionEstimates, transitionEstimates):
    """ Viterbi algorithm """
    tags = list(emissionEstimates)
    pi = [{tag: [0.0, ''] for tag in list(emissionEstimates)} for o in observationSequence]

    # Initialization
    for c_tag in tags:
        if c_tag not in transitionEstimates['##START##']: continue  # update tags which can be transitioned from ##START##

        if observationSequence[0] in m_training:  # if this word is not ##UNK##
            if observationSequence[0] in emissionEstimates[c_tag]:  # and this emission can be found
                emission = emissionEstimates[c_tag][observationSequence[0]]
            else:  # but this emission doesn't exist
                emission = 0.0
        else:  # if this word is ##UNK##
            emission = emissionEstimates[c_tag]['##UNK##']

        pi[0][c_tag] = [transitionEstimates['##START##'][c_tag] * emission, '##START##']

    # Recursive case
    for k in range(1, len(observationSequence)):  # pi[k][c_tag] = max(a(p_tag, c_tag)...)
        for c_tag in tags:
            for p_tag in tags:
                if c_tag not in transitionEstimates[p_tag]: continue  # only compare p_tags which can transition to c_tag

                score = pi[k-1][p_tag][0] * transitionEstimates[p_tag][c_tag]
                if score > pi[k][c_tag][0]:
                    pi[k][c_tag] = [score, p_tag]

            if observationSequence[k] in m_training:  # if this word is not ##UNK##
                if observationSequence[k] in emissionEstimates[c_tag]:  # and this emission can be found
                    emission = emissionEstimates[c_tag][observationSequence[k]]
                else:  # but this emission doesn't exist
                    emission = 0.0
            else:  # if this word is ##UNK##
                emission = emissionEstimates[c_tag]['##UNK##']

            pi[k][c_tag][0] *= emission

    # Finally
    result = [0.0, '']
    for p_tag in tags:
        if '##STOP##' not in transitionEstimates[p_tag]: continue  # only compare p_tags which can transition to ##STOP##

        score = pi[-1][p_tag][0] * transitionEstimates[p_tag]['##STOP##']
        if score > result[0]:
            result = [score, p_tag]

    # Backtracking
    if not result[1]:  # for those weird cases where the final probability is 0
        return

    prediction = [result[1]]
    for k in reversed(range(len(observationSequence))):
        if k == 0: break  # skip ##START## tag
        prediction.insert(0, pi[k][prediction[0]][1])

    return prediction

transitionEstimates = estimateTransition(trainFilePath)
m_training, emissionEstimates = estimateEmission(trainFilePath)
sentimentAnalysis(inputTestFilePath, m_training, emissionEstimates, transitionEstimates, outputTestFilePath)