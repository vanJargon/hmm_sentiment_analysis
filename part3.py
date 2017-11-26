import sys

DEBUG=False
def file_parse(filename, training):
    # Open the file to read
    f = open(filename, "r")

    if training:
        #initializing the return variables
        X = []
        Y = []

        #initializing the temporary variables
        wordL = []
        tagL = []

        #going over every line in the file
        for line in f:
            #repr to get the \n char, stripping the " and '.
            item = line.strip("\n\r")

            #checking for new sentence
            if len(item) != 0:
                # print(item)
                #splitting the line into word and tag
                word, tag = item.split(" ")

                # debugging stuff
                if DEBUG:
                    print(repr(line.strip("\n\r")).strip("'").strip('"'), len(repr(line.strip("\n\r")).strip("'").strip('"')))
                    print(word)

                #appending word and tag into temp variables
                wordL.append(word)
                tagL.append(tag)
            else:
                #appending temp variables to X and Y
                X.append(wordL)
                Y.append(tagL)

                #reinitializing temp variables
                wordL = []
                tagL = []

        # performing sanity check for X and Y- added a return so it breaks here
        for i in range(len(X)):
            if len(X[i]) != len(Y[i]):
                print("issue at sentence", str(i))



        return X, Y
    else:
        #initializing the return variables
        X = []

        #initializing the temporary variables
        wordL = []

        #going over every line in the file
        for line in f:
            #repr to get the \n char, stripping the " and '.
            item = line.strip("\n\r")
            
            #checking for new sentence
            if len(item) != 0:
                #splitting the line into word and tag
                word = item

                # debugging stuff
                if DEBUG:
                    print(repr(line.strip("\n\r")).strip("'").strip('"'), len(repr(line.strip("\n\r")).strip("'").strip('"')))
                    print(word)

                #appending word and tag into temp variables
                wordL.append(word)
            else:
                #appending temp variables to X and Y
                X.append(wordL)

                #reinitializing temp variables
                wordL = []

        return X

    
# Test
# print(file_parse("EN/dev.in", False))


# Assume that training data is split over two lists: X and Y
# where every X[i] contains words in one sentence
# and Y[i] contains respective tags for that sentence
# which means, X[i][j] is a word, and Y[i][j] is corresponding tag


# HELPER FUNCTIONS
# -------------------------------------
# counts number of times y appears in Y
def countY(Y, tag):
    total_Y = 0 
    for tags_for_sentence in Y:
        total_Y += tags_for_sentence.count(tag)
    return total_Y

# check if x is in X
def checkX(X, word):
    for sentence in X:
        if word in sentence:
            return True

# get all unique tags in Y
def getUniqueY(Y):
    return list(set(y for l in Y for y in l))

# get all unique words in X
def getUniqueX(X):
    return list(set(x for l in X for x in l))

# count pattern of tags
# ASSUMING THAT START = 0, STOP = 9
def countPattern(Y, pattern):
    all_Y = ''
    for y in Y:
        all_Y += '0'
        all_Y += ''.join(map(str,y))
        all_Y += '9'
    # print(all_Y)
    return all_Y.count(pattern)

# -------------------------------------------------------------------------------------------
# PART 2

# EMISSION PARAMETERS FOR ONE (xi, yi)
# -------------------------------------
# get emission estimates
def emissionParameter(X, Y, x, y):
    count_YX = 0
    # assume that training set is properly formed
    # which means len(X) == len(Y)
    # and len(X[i]) == len(Y[i]) for all i
    length = len(X)
    # if x is in training set
    if checkX(X,x):
        for i in range(0, length):
            for j in range(0, len(X[i])):
                if Y[i][j] == y and X[i][j] == x:
                    count_YX += 1
        return (count_YX/(countY(Y, y) + 1))
    # if not
    else:
        return (1/(countY(Y, y) + 1))

# CREATING AN EMISSION TABLE
# -------------------------------------------------------------------------------------------
def emissionTable(X, Y, X_test):
    emissionTable = {}
    unique_tags = getUniqueY(Y)
    unique_words = getUniqueX(X_test)

    for word in unique_words:
        for tag in unique_tags:
            emissionTable[(word, tag)] = emissionParameter(X, Y, word, tag)
    return emissionTable

# GET TAGS FOR ALL SENTENCES USING EMISSION PARAMETERS
# -------------------------------------
# Implement a simple sentiment analysis system that produces the tag
# y∗ = arg max e(x|y)
# for each word x in the sequence
# X, Y
def getTag(X_Test, X, Y):
    EMISSION = emissionTable(X_Test, X, Y)
    # dictionary of {word : tag}
    tags_for_X = {}    
    # unique tags
    unique_tags = getUniqueY(Y)
    # unique words, because here, order does not matter
    unique_words = getUniqueX(X_Test)
    print("Getting tags..   ")
    counter = 0

    for word in unique_words:
        counter += 1
        possible_Y = {}
        for tag in unique_tags:
            possible_Y[tag] = EMISSION[(word, tag)]
        # print("word: " + str(word) + ", possible y: " + str(possible_Y))
        max_val = max(possible_Y.values())
        # print("max: " + str(max(possible_Y.values())))    
        tags_for_X[word] = list(possible_Y.keys())[list(possible_Y.values()).index(max_val)]
        print("1 word down, " + str(len(unique_words) - counter) + " to go")
    print("Done!")
    return tags_for_X

# -------------------------------------------------------------------------------------------
# PART 3


# TRANSITION PARAMETERS FOR ONE (yi-1, yi)
# -------------------------------------
# transition parameter q(yi|yi-1) = count(yi-1, yi)/count(yi-1)
def transitionParameter(Y, yi_minus_one, yi):
    # print(yi_minus_one, yi)
    if yi_minus_one == 'START':
        pattern = '0' + yi
        count_yi_minus_one = len(Y)

    elif yi == 'STOP':
        pattern = yi_minus_one + '9'
        count_yi_minus_one = countY(Y, yi_minus_one)

    else:
        pattern = yi_minus_one + yi
        count_yi_minus_one = countY(Y, yi_minus_one)

    transiton_count = countPattern(Y, pattern) 
    return transiton_count/count_yi_minus_one


def transitionTable(Y):
    transitionTable = {}
    unique_tags = getUniqueY(Y)

    for tag in unique_tags:
        transitionTable[('START', tag)] = transitionParameter(Y, 'START', tag)
        transitionTable[(tag, 'STOP')] = transitionParameter(Y, tag, 'STOP')
        for next_tag in unique_tags:
            transitionTable[(tag, next_tag)] = transitionParameter(Y, tag, next_tag)
    return transitionTable

# test cases- THESE ARE BAD ONES, BUT THEY CHECK FUNCTIONALITY, SO OH WELL
# X = [["the", "cow", "jumped", "over", "the", "moon"], ["the", "dish", "ran", "away", "with", "the", "spoon"]]
# Y = [["D", "N", "V", "P", "D", "N"], ["D", "N", "V", "A", "P", "D", "N"]]
# X_Test = [["the", "cat", "cried", "over", "the", "milk"], ["the", "Spoon", "and", "fork", "ran", "away", "from", "the", "knife"]]


# print(getUniqueX(X_Test))
# print("for word in training set:" + str(emissionParameter(X, Y, "the", "D")))
# print("for word in training set:" + str(emissionParameter(X, Y, "the", "P")))
# print(getTag(X_Test, X, Y))
# print("transition params: " + str(transitionParameter(Y, 'START', 'D')))
# print("transition params: " + str(transitionParameter(Y, 'N', 'STOP')))

# PART 3
# ----------------------------------------------------
# Viterbi algorithm 

# FOR ONE SENTENCE YA
# N is number of words in 1 sentence
# sentence is sentence
# sentence[k-1] is word at pos k-1 in THAT PARTICULAR SENTENCE
# EMISSION_TABLE is emission table
# TRANSITION_TABLE is transition table
def viterbi(sentence, N, TRANSITION_TABLE, EMISSION_TABLE, unique_tags):
    if(sentence == []):
        return "NULL"
    trellis = dict.fromkeys(list(range(1, N+2)))
    # 1 to 
    for k in range(1, N+1):
        trellis[k] = dict.fromkeys(unique_tags)

        # for the first layer, the pi(k-1, v) is the start state.
        if k == 1:
            word_to_emit = sentence[k-1]
            for tag in unique_tags:
                # trellis[k][tag] = (1*emissionParameter(X, Y, word_to_emit, tag)*transitionParameter(Y, 'START', tag), 'START')
                trellis[k][tag] = (1*EMISSION_TABLE[(word_to_emit, tag)]*TRANSITION_TABLE[('START', tag)], 'START')

        
        # for everything else
        else:
            word_to_emit = sentence[k-1]
            for tag in unique_tags:
                possible_tags = dict.fromkeys(unique_tags)
           
                for prev_tag in unique_tags:
                    possible_tags[prev_tag] = trellis[k-1][prev_tag][0]*EMISSION_TABLE[(word_to_emit, tag)]*TRANSITION_TABLE[(prev_tag, tag)]
                trellis[k][tag] = (max(possible_tags.values()), list(possible_tags.keys())[list(possible_tags.values()).index(max(possible_tags.values()))])
        # print("trellis: " + str(trellis[k]))
    
    # stop case
    possible_tags = dict.fromkeys(unique_tags)
    for tag in unique_tags:
        possible_tags[tag] = trellis[N][tag][0]*TRANSITION_TABLE[(tag, 'STOP')]
    trellis[N+1] = {'STOP': (max(possible_tags.values()), list(possible_tags.keys())[list(possible_tags.values()).index(max(possible_tags.values()))])}

    return backtrack(trellis) 
    

def backtrack(trellis):
    N = len(trellis.keys())
    path = [0]*N

    path[N-1] = 'STOP'

    for i in range(0, (N-1)):
        path[N-i-2] = trellis[N-i][path[N-1-i]][1]
    return path



# TEST CASES:
# X = [["the", "cow", "jumped", "over", "the", "moon"], ["the", "dish", "ran", "away", "with", "the", "spoon"]]
# Y = [["D", "N", "V", "P", "D", "N"], ["D", "N", "V", "A", "P", "D", "N"]]
# X_Test = [["the", "cat", "cried", "over", "the", "milk"], ["the", "Spoon", "and", "fork", "ran", "away", "from", "the", "knife"]]
# -----------------------------------------------------------------------------------------
# X = [["b", "c", "a", "b"], ["a", "b", "a"], ["b", "c", "a", "b", "c"], ["c", "b", "a"]]
# Y = [["X", "Y", "Z", "X"], ["X", "Z", "Y"], ["Z", "Y", "X", "Z", "Y"], ["Z", "X", "Y"]]
# X_test = [["b", "b"]]
# EMISSION = emissionTable(X, Y, X_test)
# TRANSITION = transitionTable(Y)
# unique_tags = getUniqueY(Y)
# print(EMISSION)
# print(TRANSITION)
# print(viterbi(X_test[0], len(X_test[0]), TRANSITION, EMISSION, unique_tags))



def getScores(outfile_generated, outfile_given):
    # DO THINGS
        return None
if __name__ == '__main__':
    train_filename = "train"
    test_filename = "dev.in"
    outfile = "output"
    function = 'viterbi'

    print(train_filename, test_filename, outfile)

    train_parsed_file = file_parse(train_filename, True)
    train_X = train_parsed_file[0]
    train_Y = train_parsed_file[1]

    test_X = file_parse(test_filename, False)

    if(function == 'getTags'):
        tags = getTag(test_X, train_X, train_Y)
                
        f = open(outfile,'w')
        for i in range(0, len(test_X)):
            for j in range(0, len(test_X[i])):
                towrite = str(test_X[i][j]) + " " + str(tags[test_X[i][j]])
                f.write(towrite+'\n')
            f.write('\n') 
        f.close() 

    if(function == 'viterbi'):
        f = open(outfile,'w')
        print("generating tables..")
        EMISSION = emissionTable(train_X, train_Y, test_X)
        print("emission done")
        TRANSITION = transitionTable(train_Y)
        print("transition done")
        unique_tags = getUniqueY(train_Y)
        print("unique tags gotten from text")
        print("All pre-requisites done, now running viterbi")
        for i in range(0, len(test_X)):
            # print(test_X[i])
            print("Writing one sentence, " + str(len(test_X)-i) + " to go.")
            viterbi_sentence = viterbi(test_X[i], len(test_X[i]), TRANSITION, EMISSION, unique_tags)
            for j in range(0, len(test_X[i])):
                towrite = str(test_X[i][j]) + " " + str(viterbi_sentence[j])
                f.write(towrite+'\n') 
            f.write('\n')
        f.close() 




