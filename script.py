#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:21:14 2017

@author: 1001827
"""
filePath = '../EN/train'
#filePath = '../FR/train'

# Part 2(a)
def estimateEmission(filePath):
    tags = {}
    observations = {}
    estimates = {}
    
    # Process the data
    for line in open(filePath, 'r'):
        segmentedLine = line.rstrip()
        if segmentedLine: # if its not just an empty string
            segmentedLine = segmentedLine.split(' ')
            
            observation = segmentedLine[0] # X
            tag = segmentedLine[1] # Y
            
            if tag not in tags: # if this tag has never been seen before
                tags[tag] = 1
                observations[tag] = {observation:1}
                
            else: # if this tag has been seen before
                tags[tag] += 1
                if observation not in observations[tag]:
                    observations[tag][observation] = 1
                else:
                    observations[tag][observation] += 1
         
    # Compute the MLE based on the count data
    for tag in tags:
        estimates[tag] = {}
        for observation in observations[tag]:
            estimates[tag][observation] = float(observations[tag][observation])/tags[tag]
    
    return estimates
                                        
print estimateEmission(filePath)