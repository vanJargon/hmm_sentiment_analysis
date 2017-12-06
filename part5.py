#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Code for 01.112 Machine Learning Project (Part 5)

Done by:
Vanessa Tan (1001827)
Shruthi Shangar (1001630)
"""
import argparse

def getGlobalFeatures(filePath, k=1):
    observations = {}
    tags = []
    emissionFeatures = {}
    transitionFeatures = {}

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
                tags.append(tag)

    # Replace observations which appear for less than k times with ##UNK##
    observationsList = list(observations)

    for observation in observations:
        if observations[observation] < k:
            observationsList.remove(observation)

    observationsList.append('##UNK##')

    print ('Generating possible emission features...')
    for tag in tags:
        emissionFeatures[tag] = {}
        for observation in observationsList:
            emissionFeatures[tag][observation] = [0.0, 0.0]  # first term is weight, second term is average weight

    print ('Generating possible transition features...')
    p_tagsList = list(tags)
    p_tagsList.append('##START##')
    c_tagsList = list(tags)
    c_tagsList.append('##STOP##')

    for p_tag in p_tagsList:
        transitionFeatures[p_tag] = {}
        for c_tag in c_tagsList:
            transitionFeatures[p_tag][c_tag] = [0.0, 0.0]

    return observationsList, emissionFeatures, transitionFeatures


def updateWeights(observationSequence, goldTags, predictedTags, m_training, emissionFeatures, transitionFeatures):
    """ Helper function to update weights """
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
                emissionFeatures[goldTags[i]][observationSequence[i]][0] += 1
                emissionFeatures[goldTags[i]][observationSequence[i]][1] += emissionFeatures[goldTags[i]][observationSequence[i]][0]

                emissionFeatures[predictedTags[i]][observationSequence[i]][0] -= 1
                emissionFeatures[predictedTags[i]][observationSequence[i]][1] += emissionFeatures[predictedTags[i]][observationSequence[i]][0]
            else:  # if this word is ##UNK##
                emissionFeatures[goldTags[i]]['##UNK##'][0] += 1
                emissionFeatures[goldTags[i]]['##UNK##'][1] += emissionFeatures[goldTags[i]]['##UNK##'][0]

                emissionFeatures[predictedTags[i]]['##UNK##'][0] -= 1
                emissionFeatures[predictedTags[i]]['##UNK##'][1] += emissionFeatures[predictedTags[i]]['##UNK##'][0]

            # Update weights for transition features
            transitionFeatures[goldTags[i-1]][goldTags[i]][0] += 1
            transitionFeatures[goldTags[i-1]][goldTags[i]][1] += transitionFeatures[goldTags[i-1]][goldTags[i]][0]
            transitionFeatures[goldTags[i]][goldTags[i+1]][0] += 1
            transitionFeatures[goldTags[i]][goldTags[i+1]][1] += transitionFeatures[goldTags[i]][goldTags[i+1]][0]

            transitionFeatures[predictedTags[i-1]][predictedTags[i]][0] -= 1
            transitionFeatures[predictedTags[i-1]][predictedTags[i]][1] += transitionFeatures[predictedTags[i-1]][predictedTags[i]][0]
            transitionFeatures[predictedTags[i]][predictedTags[i+1]][0] -= 1
            transitionFeatures[predictedTags[i]][predictedTags[i+1]][1] += transitionFeatures[predictedTags[i]][predictedTags[i+1]][0]

    return emissionFeatures, transitionFeatures


def viterbi(observationSequence, m_training, emissionFeatures, transitionFeatures):
    """ Viterbi algorithm """
    tags = list(emissionFeatures)
    pi = [{tag: [None, ''] for tag in tags} for o in observationSequence]

    # Initialization
    for c_tag in tags:
        score = 0.0
        # account for transition features
        score += transitionFeatures['##START##'][c_tag][0]

        # account for emission features
        if observationSequence[0] in m_training:  # if this word is not ##UNK##
            score += emissionFeatures[c_tag][observationSequence[0]][0]
        else:  # if this word is ##UNK##
            score += emissionFeatures[c_tag]['##UNK##'][0]

        pi[0][c_tag] = [score, '##START##']

    # Recursive case
    for k in range(1, len(observationSequence)):  # pi[k][c_tag] = max(a(p_tag, c_tag)...)
        for c_tag in tags:
            for p_tag in tags:
                score = pi[k-1][p_tag][0]
                # account for transition features
                score += transitionFeatures[p_tag][c_tag][0]

                if score > pi[k][c_tag][0]:
                    pi[k][c_tag] = [score, p_tag]

            # Since the emission score is not dependent on p_tag, we add it in outside the previous loop
            if observationSequence[k] in m_training:  # if this word is not ##UNK##
                pi[k][c_tag][0] += emissionFeatures[c_tag][observationSequence[k]][0]
            else:  # if this word is ##UNK##
                pi[k][c_tag][0] += emissionFeatures[c_tag]['##UNK##'][0]

    # Finally
    result = [None, '']
    for p_tag in tags:
        # account for final transition to '##STOP##'
        score = pi[-1][p_tag][0] + transitionFeatures[p_tag]['##STOP##'][0]

        if score > result[0]:
            result = [score, p_tag]

    # Backtracking
    prediction = [result[1]]
    for k in reversed(range(len(observationSequence))):
        if k == 0: break  # skip ##START## tag
        prediction.insert(0, pi[k][prediction[0]][1])

    return prediction


def trainModel(filePath, m_training, emissionFeatures, transitionFeatures, numIters=4):  # FR tried till numIters=22
    """ trains the model according to the structured perceptron algorithm """
    n = 0  # number of instances in training (nT)

    # Iterate over training data for numIters
    for t in range(numIters):
        print 'Training model in iteration', t, '...'
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
                # Update the parameters according to the differences between predicted and gold tag sequence
                emissionFeatures, transitionFeatures = updateWeights(observationSequence, goldTagSequence, predictionTagSequence, m_training, emissionFeatures, transitionFeatures)

                n += 1
                observationSequence = []
                goldTagSequence = []

    # Calculate the average weights for emission features
    for tag in list(emissionFeatures):
        for observation in emissionFeatures[tag]:
            emissionFeatures[tag][observation][0] /= (n+1)

    # Calculate the average weights for transition features
    for p_tag in list(transitionFeatures):
        for c_tag in transitionFeatures[p_tag]:
            transitionFeatures[p_tag][c_tag][0] /= (n+1)

    return emissionFeatures, transitionFeatures


def sentimentAnalysis(inputPath, m_training, emissionEstimates, transitionEstimates, outputPath):
    """ splits test file into separate observation sequences and feeds them into Viterbi algorithm """
    f = open(outputPath, 'w')

    observationSequence = []
    proc_observationSequence = []
    for line in open(inputPath, 'r'):
        observation = line.rstrip()
        if observation:
            observationSequence.append(observation)
            proc_observationSequence.append(observation.lower())
        else:
            predictionSequence = viterbi(proc_observationSequence, m_training, emissionEstimates, transitionEstimates)
            for i in range(len(observationSequence)):
                f.write('%s %s\n' % (observationSequence[i], predictionSequence[i]))
            f.write('\n')
            observationSequence = []
            proc_observationSequence = []

    print 'Finished writing to file %s' % (outputPath)
    return f.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, dest='dataset', help='Dataset to run script over', required=True)
    parser.add_argument('-k', type=int, dest='k', help='Minimum number of times a word needs to appear to not be replaced', default=1, required=False)
    parser.add_argument('-i', type=int, dest='i', help='Number of iterations over training data in Structured Perceptron algorithm', default=4, required=False)

    args = parser.parse_args()

    trainFilePath = '../%s/train' % (args.dataset)
    inputTestFilePath = '../%s/dev.in' % (args.dataset)
    outputTestFilePath = '../%s/dev.p5.out' % (args.dataset)
    inputHiddenTestFilePath = '../%s/test.in' % (args.dataset)
    outputHiddenTestFilePath = '../%s/test.p5.out' % (args.dataset)

    m_training, emissionFeatures, transitionFeatures = getGlobalFeatures(trainFilePath, k=args.k)
    emissionFeatures, transitionFeatures = trainModel(trainFilePath, m_training, emissionFeatures, transitionFeatures, numIters=args.i)
    sentimentAnalysis(inputTestFilePath, m_training, emissionFeatures, transitionFeatures, outputTestFilePath)
    sentimentAnalysis(inputHiddenTestFilePath, m_training, emissionFeatures, transitionFeatures, outputHiddenTestFilePath)