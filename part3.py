from heapq import nlargest
from math import log

Input_data = 'dev.in'
Output_data = 'dev.out'
Train_data= 'train'


def readTraining(filename):
    result = []
    fo = open(filename, 'r')
    while True:
        line = fo.readline()
        if(line == ''):
            break
        if(line == '\n'):
            result.append((None, None))
            continue
        temp = line.replace('\n','').split(' ')
        result.append(tuple(temp))

    fo.close()
    return result

def readTesting(filename):
    result = []
    fo = open(filename, 'r')
    while True:
        line = fo.readline()
        if(line == ''):
            break
        if(line == '\n'):
            result.append(None)
            continue
        result.append(line.replace('\n',''))

    fo.close()
    return result

def writeResult(filename, result):
    fo = open(filename, 'w')
    for item in result:
        if item == (None, None):
            fo.write('\n')
            continue
        fo.write(item[0])
        fo.write(' ')
        fo.write(item[1])
        fo.write('\n')
    fo.close()
def calculate_part3(filename_train, filename_test, part4 = False, part5 = False):
    max_func = lambda v: max(v, key = lambda x: x[0])
    train = readTraining(filename_train)

    # Build transition probability
    # Count every tags and transitions
    count = {'_START': 0, '_STOP': 0}
    trans = {}
    for i in range(len(train)):
        if i == 0: #2 ways to start
            count['_START'] += 1
        elif train[i - 1] == (None, None):
            count['_START'] += 1

        if train[i] == (None, None): # End of sentence
            count['_STOP'] += 1
            if train[i-1][1] not in trans: #if not trans.has_key(train[i - 1][1]):
                trans[train[i - 1][1]] = {}
            if '_STOP' not in trans[train[i-1][1]]:#if not trans[train[i - 1][1]].has_key('_STOP'):
                trans[train[i - 1][1]]['_STOP'] = 0
            trans[train[i - 1][1]]['_STOP'] += 1
        else: # A word
            if train[i][1] not in count: #if not count.has_key(train[i][1]):
                count[train[i][1]] = 0
            count[train[i][1]] += 1
            if i == 0: # Start
                if '_START' not in trans:
                    trans['_START'] = {}
                if train[i][1] not in trans['_START']:
                    trans['_START'][train[i][1]] = 0
                trans['_START'][train[i][1]] += 1
            if train[i - 1] == (None, None): # Start
                if ('_START') not in trans:
                    trans['_START'] = {}
                if train[i][1] not in trans['_START']:  
                    trans['_START'][train[i][1]] = 0
                trans['_START'][train[i][1]] += 1
            else: # In the middle of sentence
                if train[i-1][1] not in trans: #if not trans.has_key(train[i - 1][1]):
                    trans[train[i - 1][1]] = {}
                if train[i][1] not in trans[train[i-1][1]]: #if not trans[train[i - 1][1]].has_key(train[i][1]):
                    trans[train[i - 1][1]][train[i][1]] = 0
                trans[train[i - 1][1]][train[i][1]] += 1

    # Calculate transition probability
    transition = {}
    for key1 in count:
        if key1 not in transition: #if not transition.has_key(key1):
            transition[key1] = {}
        for key2 in count:
            if key2 not in transition[key1]: #if not transition[key1].has_key(key2):
                transition[key1][key2] = float('-inf')
            if key1 in trans: #if trans.has_key(key1):
                if key2 in trans[key1]: #if trans[key1].has_key(key2):
                    transition[key1][key2] = log(trans[key1][key2]) - log(count[key1])

    # Count emissions
    em = {}
    for item in train:
        if item == (None, None):
            continue
        if item[1] not in em: #if not em.has_key(item[1]):
            em[item[1]] = {}
        if item[0] not in em[item[1]]: #if not em[item[1]].has_key(item[0]):
            em[item[1]][item[0]] = 0
        em[item[1]][item[0]] += 1

    # Calculate emission probability
    emission = {}
    for tag in em:
        if tag not in emission: #if not emission.has_key(tag):
            emission[tag] = {}
        for word in em[tag]:
            emission[tag][word] = log(em[tag][word]) - log(count[tag] + 1)

    # Viterbi
    test = readTesting(filename_test)
    result = []
    sentence = []
    for w in test:
        if(w != None):
            sentence.append(w)
        else: # End of a sentence
            # Initialize viterbi
            viterbi = [{} for i in range(len(sentence) + 1)]
            for k in viterbi:
                for tag in count:
                    if count == '_STOP': # Skip STOP tag
                        continue
                    k[tag] = (float('-inf'), None)
            viterbi[0]['_START'] = (0.0, None)

            n = len(viterbi) - 1
            for k in range(1, n + 1):
                found = False
                for emission_tag in emission:
                    if sentence[k-1] not in emission[emission_tag]: #if(emission[emission_tag].has_key(sentence[k - 1])):
                        found = True

                for tag in count:
                    if tag == '_START' or tag == '_STOP':
                        viterbi[k][tag] = (float('-inf'), None)
                        continue
                    values = []
                    for prev_tag in count:
                        if tag == '_STOP':
                            continue
                        temp = viterbi[k - 1][prev_tag][0] # pi(k-1, u)
                        temp += transition[prev_tag][tag] # * a(u, v)
                        # * b(v, x_k)
                        if tag in emission: #if emission.has_key(tag):
                            if sentence[k-1] not in emission[tag]: #if emission[tag].has_key(sentence[k - 1]):
                                temp += emission[tag][sentence[k - 1]]
                            else:
                                if found:
                                    temp += float('-inf')
                                else:
                                    if part5:
                                        temp += log(count[tag]) - log(count[tag] + 1)
                                    else:
                                        temp += (-log(count[tag] + 1))
                        else:
                            if found:
                                temp += float('-inf')
                            else:
                                if part5:
                                    temp += log(count[tag]) - log(count[tag] + 1)
                                else:
                                    temp += (-log(count[tag] + 1))
                        values.append((temp, prev_tag))

                    allzero = True
                    for v in values:
                        if v[0] != float('-inf'):
                            allzero = False
                    if not allzero:
                        viterbi[k][tag] = max_func(values)
                    else:
                        viterbi[k][tag] = (float('-inf'), None)

                blocked = True
                for tag in viterbi[k]:
                    if viterbi[k][tag][0] != float('-inf'):
                        blocked = False
                if blocked:
                    for tag in count:
                        if tag == '_START' or tag == '_STOP':
                            viterbi[k][tag] = (float('-inf'), None)
                            continue
                        values = []
                        for prev_tag in count:
                            if tag == '_STOP':
                                continue
                            temp = viterbi[k - 1][prev_tag][0] # pi(k-1, u)
                            # * a(u, v) = 1 (Blocked mode)
                            # * b(v, x_k)
                            if tag in emission: #if emission.has_key(tag):
                                if sentence[k-1] not in emission[tag]: #if emission[tag].has_key(sentence[k - 1]):
                                    temp += emission[tag][sentence[k - 1]]
                                else:
                                    if found:
                                        temp += float('-inf')
                                    else:
                                        if part5:
                                            temp += log(count[tag]) - log(count[tag] + 1)
                                        else:
                                            temp += (-log(count[tag] + 1))
                            else:
                                if found:
                                    temp += float('-inf')
                                else:
                                    if part5:
                                        temp += log(count[tag]) - log(count[tag] + 1)
                                    else:
                                        temp += (-log(count[tag] + 1))
                            values.append((temp, prev_tag))

                        allzero = True
                        for v in values:
                            if v[0] != float('-inf'):
                                allzero = False
                        if not allzero:
                            viterbi[k][tag] = max_func(values)
                        else:
                            viterbi[k][tag] = (float('-inf'), None)

            values = []
            for tag in viterbi[n]: # STOP
                values.append((viterbi[n][tag][0] + transition[tag]['_STOP'], tag))
            # Check if all zero
            allzero = True
            for v in values:
                if v[0] != float('-inf'):
                    allzero = False
            if allzero:
                values = []
                for tag in viterbi[n]: # STOP (Blocked mode)
                    values.append((viterbi[n][tag][0], tag))
            if not part4:
                stop_value = max_func(values)
            else:
                ten_values = nlargest(10, values, key = lambda x: x[0])
                for i in range(len(ten_values) - 1, -1, -1):
                    if ten_values[i][0] == float('-inf'):
                        continue
                    else:
                        stop_value = ten_values[i]
                        break

            # Backtracking
            result_tags = [stop_value[1]]
            for i in range(n, 0, -1):
                if viterbi[i][result_tags[0]][1] != '_START':
                    result_tags.insert(0, viterbi[i][result_tags[0]][1])
                else:
                    break

            for i in range(len(sentence)):
                result.append((sentence[i], result_tags[i]))
            result.append((None, None))
            # Prepare for next sentence
            sentence = []
    return result

def part3():
    result = calculate_part3(Train_data, Input_data)
    writeResult(Result_data, result)
    correct = readTraining(Output_data)
    acc = accuracy(result, correct)
    if acc != None:
        print ('[Part 3] Accuracy:', acc)

    
        


if __name__ == '__main__':
    pass
    #part2()
    part3()
    #part4()
    #part5()
    #test()
