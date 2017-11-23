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
outputTestFilePath = '../%s/dev.p2.out' % (dataset)


# Part 2
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

    # Compute the MLE based on the count data and the number of times the observation appears in training data
    for tag in tags:
        estimates[tag] = {}
        for observation in l_Observations[tag]:
            if observations[observation] >= k:
                estimates[tag][observation] = float(l_Observations[tag][observation]) / tags[tag]

    # print tags
    # print observations
    # print l_Observations
    # print estimates
    return estimates


def sentimentAnalysis(inputPath, estimates, outputPath):
    f = open(outputPath, 'w')
    for line in open(inputPath, 'r'):
        observation = line.rstrip()
        if observation:
            prediction = ('', 0.0)  # prediction is tuple of tag and the MLE of observation for the given tag
            for tag in estimates:
                if observation in estimates[tag] and estimates[tag][observation] > prediction[1]:
                    prediction = (tag, estimates[tag][observation])
            if prediction[0]:
                f.write('%s %s\n' % (observation, prediction[0]))
            else:
                f.write('##UNK##\n')
        else:
            f.write('\n')

    print 'Finished writing to file %s...' % (outputPath)
    return f.close()


estimates = estimateEmission(trainFilePath)
sentimentAnalysis(inputTestFilePath, estimates, outputTestFilePath)