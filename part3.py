#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:21:14 2017

@author: 1001827
"""
#from script import estimateEmission

dataset = 'EN'
# dataset = 'FR'
# dataset = 'CN'
# dataset = 'SG'

trainFilePath = '../%s/train' % (dataset)
inputTestFilePath = '../%s/dev.in' % (dataset)
outputTestFilePath = '../%s/dev.p3.out' % (dataset)

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

def sentimentAnalysis(inputPath, emissionEstimates, transitionEstimates, outputPath):
    ''' splits test file into separate observation sequences and feeds them into Viterbi algorithm '''
    observationSequence = []
    for line in open(inputPath, 'r'):
        observation = line.rstrip()
        if observation:
            observationSequence += observation
        else:
            viterbi(observationSequence, emissionEstimates, transitionEstimates)
            observationSequence = []

def viterbi(observationSequence, emissionEstimates, transitionEstimates):
    ''' recursive viterbi algorithm '''
    

print estimateTransition(trainFilePath)




