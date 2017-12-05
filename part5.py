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


def countEmissions(filePath, k=3):
    observations = {}
    l_Observations = {}  # count of labelled observations

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

            if tag not in l_Observations:  # if this tag has never been seen before
                l_Observations[tag] = {observation: [0.0, 0.0, 1]}  # first term is weighted_average, second term is weight, third term is count

            else:  # if this tag has been seen before
                if observation not in l_Observations[tag]:
                    l_Observations[tag][observation] = [0.0, 0.0, 1]
                else:
                    l_Observations[tag][observation][2] += 1

    # Replace observations which appear for less than k times with ##UNK##
    for tag in l_Observations:
        l_Observations[tag]['##UNK##'] = [0.0, 0.0, 0]
        for observation in list(l_Observations[tag]):  # loop over all keys in l_Observations
            if observation == '##UNK##': continue

            if observation not in observations:  # if this observation has been found to appear less than k times before
                l_Observations[tag]['##UNK##'][2] += l_Observations[tag].pop(observation)[2]
            elif observations[observation] < k:  # or if first meet an observation that appear less than k times
                l_Observations[tag]['##UNK##'][2] += l_Observations[tag].pop(observation)[2]
                del observations[observation]

    return list(observations), l_Observations


def countTransitions(filePath):
    t_Tags = {}  # count the number of times a particular transition from y(i) to y(i-1) has appeared

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

        if previousState not in t_Tags:  # if tag y(i-1) has never been seen before
            t_Tags[previousState] = {currentState: [0.0, 0.0, 1]}  # first term is weighted_average, second term is weight, third term is count
        else:
            if currentState not in t_Tags[previousState]:
                t_Tags[previousState][currentState] = [0.0, 0.0, 1]
            else:
                t_Tags[previousState][currentState][2] += 1

    return t_Tags


def updateWeights(observationSequence, goldTags, predictedTags, m_training, emissionFeatures, transitionFeatures):
    """ Helper function to update weights """
    # print 'gold:', goldTags
    # print 'pred:', predictedTags

    goldTags.insert(0, '##START##')
    goldTags.append('##STOP##')
    predictedTags.insert(0, '##START##')
    predictedTags.append('##STOP##')
    observationSequence.insert(0, '')
    observationSequence.append('')

    for i in range(len(goldTags)):
        if goldTags[i] != predictedTags[i]:
            # Update weights for emission features
            if observationSequence[i] in m_training:  # if this word is not ##UNK##
                emissionFeatures[goldTags[i]][observationSequence[i]][1] += 1
                emissionFeatures[goldTags[i]][observationSequence[i]][0] += emissionFeatures[goldTags[i]][observationSequence[i]][1]
                if observationSequence[i] in emissionFeatures[predictedTags[i]]:
                    emissionFeatures[predictedTags[i]][observationSequence[i]][1] -= 1
                    emissionFeatures[predictedTags[i]][observationSequence[i]][0] += emissionFeatures[predictedTags[i]][observationSequence[i]][1]
            else:  # if this word is ##UNK##
                emissionFeatures[goldTags[i]]['##UNK##'][1] += 1
                emissionFeatures[goldTags[i]]['##UNK##'][0] += emissionFeatures[goldTags[i]]['##UNK##'][1]
                emissionFeatures[predictedTags[i]]['##UNK##'][1] -= 1
                emissionFeatures[predictedTags[i]]['##UNK##'][0] += emissionFeatures[predictedTags[i]]['##UNK##'][1]

            # Update weights for transition features
            transitionFeatures[goldTags[i-1]][goldTags[i]][1] += 1
            transitionFeatures[goldTags[i - 1]][goldTags[i]][0] += transitionFeatures[goldTags[i-1]][goldTags[i]][1]
            transitionFeatures[goldTags[i]][goldTags[i+1]][1] += 1
            transitionFeatures[goldTags[i]][goldTags[i + 1]][0] += transitionFeatures[goldTags[i]][goldTags[i+1]][1]
            if predictedTags[i] in transitionFeatures[predictedTags[i-1]]:
                transitionFeatures[predictedTags[i-1]][predictedTags[i]][1] -= 1
                transitionFeatures[predictedTags[i - 1]][predictedTags[i]][0] += transitionFeatures[predictedTags[i-1]][predictedTags[i]][1]
            if predictedTags[i + 1] in transitionFeatures[predictedTags[i]]:
                transitionFeatures[predictedTags[i]][predictedTags[i + 1]][1] -= 1
                transitionFeatures[predictedTags[i]][predictedTags[i + 1]][0] += transitionFeatures[predictedTags[i]][predictedTags[i + 1]][1]
    return emissionFeatures, transitionFeatures


def viterbi(observationSequence, m_training, emissionFeatures, transitionFeatures):
    """ Viterbi algorithm """
    tags = list(emissionFeatures)
    pi = [{tag: [None, ''] for tag in tags} for o in observationSequence]

    # Initialization
    for c_tag in tags:
        score = 0.0
        # account for transition features
        if c_tag in transitionFeatures['##START##']:
            # score += transitionFeatures['##START##'][c_tag][1] * transitionFeatures['##START##'][c_tag][2]
            score += transitionFeatures['##START##'][c_tag][1]

        # account for emission features
        if observationSequence[0] in m_training:  # if this word is not ##UNK##
            if observationSequence[0] in emissionFeatures[c_tag]:  # add if this emission can be found
                # score += emissionFeatures[c_tag][observationSequence[0]][1] * emissionFeatures[c_tag][observationSequence[0]][2]
                score += emissionFeatures[c_tag][observationSequence[0]][1]
        else:  # if this word is ##UNK##
            # score += emissionFeatures[c_tag]['##UNK##'][1] * emissionFeatures[c_tag]['##UNK##'][2]
            score += emissionFeatures[c_tag]['##UNK##'][1]

        pi[0][c_tag] = [score, '##START##']

    # Recursive case
    for k in range(1, len(observationSequence)):  # pi[k][c_tag] = max(a(p_tag, c_tag)...)
        for c_tag in tags:
            for p_tag in tags:
                score = pi[k-1][p_tag][0]
                # account for transition features
                if c_tag in transitionFeatures[p_tag]:
                    # score += transitionFeatures[p_tag][c_tag][1] * transitionFeatures[p_tag][c_tag][2]
                    score += transitionFeatures[p_tag][c_tag][1]

                if score > pi[k][c_tag][0]:
                    pi[k][c_tag] = [score, p_tag]

            # Since the emission score is not dependent on p_tag, we add it in outside the previous loop
            if observationSequence[k] in m_training:  # if this word is not ##UNK##
                if observationSequence[k] in emissionFeatures[c_tag]:  # and this emission can be found
                    # pi[k][c_tag][0] += emissionFeatures[c_tag][observationSequence[k]][1] * emissionFeatures[c_tag][observationSequence[k]][2]
                    pi[k][c_tag][0] += emissionFeatures[c_tag][observationSequence[k]][1]
            else:  # if this word is ##UNK##
                # pi[k][c_tag][0] += emissionFeatures[c_tag]['##UNK##'][1] * emissionFeatures[c_tag]['##UNK##'][2]
                pi[k][c_tag][0] += emissionFeatures[c_tag]['##UNK##'][1]
        # print pi[k]

    # Finally
    result = [None, '']
    for p_tag in tags:
        score = pi[-1][p_tag][0]

        # account for final transition score to '##STOP##'
        if '##STOP##' in transitionFeatures[p_tag]:
            # score += transitionFeatures[p_tag]['##STOP##'][1] * transitionFeatures[p_tag]['##STOP##'][2]
            score += transitionFeatures[p_tag]['##STOP##'][1]
        if score > result[0]:
            result = [score, p_tag]

    # Backtracking
    prediction = [result[1]]
    for k in reversed(range(len(observationSequence))):
        if k == 0: break  # skip ##START## tag
        prediction.insert(0, pi[k][prediction[0]][1])

    return prediction


def trainModel(filePath, m_training, emissionFeatures, transitionFeatures, numIters=13):
    """ trains the model according to the structured perceptron algorithm """
    tags = list(emissionFeatures)
    n = 0  # number of instances in training (nT)

    # Iterate over training data for numIters
    for t in range(numIters):
        print 'Iteration:', t
        observationSequence = []
        goldTagSequence = []
        for line in open(filePath, 'r'):
            segmentedLine = line.rstrip()
            if segmentedLine:  # if its not just an empty string
                segmentedLine = segmentedLine.rsplit(' ', 1)
                observation = segmentedLine[0]  # X
                tag = segmentedLine[1]  # gold Y
                observationSequence.append(observation.lower())
                goldTagSequence.append(tag)
            else:
                # Using the parameters, run Viterbi
                predictionTagSequence = viterbi(observationSequence, m_training, emissionFeatures, transitionFeatures)
                # print 'pred:', predictionTagSequence
                # print 'gold:', goldTagSequence
                # print '\n'
                # Update the parameters according to the differences between predicted and gold tag sequence
                emissionFeatures, transitionFeatures = updateWeights(observationSequence, goldTagSequence, predictionTagSequence, m_training, emissionFeatures, transitionFeatures)

                n += 1
                observationSequence = []
                goldTagSequence = []

    # # Calculate the average weights for emission features
    # for tag in tags:
    #     for observation in emissionFeatures[tag]:
    #         emissionFeatures[tag][observation][1] = emissionFeatures[tag][observation][0]/n
    #
    # # Calculate the average weights for transition features
    # for p_tag in tags:
    #     for c_tag in transitionFeatures[p_tag]:
    #         transitionFeatures[p_tag][c_tag][1] = transitionFeatures[p_tag][c_tag][0]/n

    return emissionFeatures, transitionFeatures

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
                f.write('%s %s\n' % (observationSequence[i], predictionSequence[i]))
            f.write('\n')
            observationSequence = []

    print 'Finished writing to file %s' % (outputPath)
    return f.close()


# get global features and initialize parameters (weights) to 0
m_training, emissionFeatures = countEmissions(trainFilePath, k=1)
transitionFeatures = countTransitions(trainFilePath)
emissionFeatures, transitionFeatures = trainModel(trainFilePath, m_training, emissionFeatures, transitionFeatures)
print emissionFeatures
print transitionFeatures
sentimentAnalysis(inputTestFilePath, m_training, emissionFeatures, transitionFeatures, outputTestFilePath)



# observation = ['Great', 'atmoshere', 'and', 'worth', 'every', 'bit', '.']
# m_training, emissionFeatures = countEmissions(trainFilePath)
# transitionFeatures = countTransitions(trainFilePath)
# print viterbi(observation,m_training,emissionFeatures,transitionFeatures)