########################################
## CS447 Natural Language Processing  ##
##           Homework 1               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
import math


def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus to help avoid sess the corpus to help avoid sparsity
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    # unigram model: unsmoothed unigram model
    # smoothed unigram model: add one smoothing with unigram model
    #  bigram model:  unsmoothed bigram model
    def __init__(self, corpus):
        self.wordsCounter=defaultdict(float)
        self.wordsTotal=0.0
        #  bigram model
        self.BiwordsCounter=defaultdict(float)
        self.trainmodel(corpus)
        self.sentenceMarkers=self.startCounter(corpus)



    #enddef
    def trainmodel(self,corpus):
        for sentence in corpus:
            for word in sentence:
                if word==start:
                    continue
                self.wordsCounter[word]+=1.0
                self.wordsTotal+=1.0

        # bigram
        prior=start
        for sentence in corpus:
            for word in sentence:
                if word==start:
                    # here the prior would be none, which will bring up the calculation fault following
                    prior=start
                    continue
                self.BiwordsCounter[word+" "+prior]+=1.0
                prior=word

    # count the start as the sentence markers
    def startCounter(self,corpus):
        # sentencemarkers: sM
        sM=0.0
        for sentence in corpus:
            for word in sentence:
                if word==start:
                    sM+=1.0
        return sM


    # Generate a sentence by drawing words according to the
    # model's probability distribution
    # Note: think about how to set the length of the sentence
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #enddef

    # return the probability of the words in the distribution
    # word/total --> unigram model
    def prob(self,word):
        return self.wordsCounter[word]/self.wordsTotal


    # return the probability of the words in the smoothed unigram
    # word+1/ total+type
    def sm_prob(self,word):
        return (self.wordsCounter[word]+1)/(self.wordsTotal+len(self.wordsCounter))

     # return the probablity of bigram mdoel
    # P(X|Y)/P(Y)
    def biprob(self,word,prior):
        return self.BiwordsCounter[word + " " + prior] / self.wordsCounter[prior]

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.wordsCounter.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

    def sm_draw(self):
        rand = random.random()
        for word in self.wordsCounter.keys():
            rand -= self.sm_prob(word)
            if rand <= 0.0:
                return word

    # Generate a bigram random word
    def drawBi(self, prior):
        rand = random.random()
        for word in self.wordsCounter.keys():
            rand -= self.biprob(word, prior)
            if rand <= 0.0:
                return word

    def drawBi_start(self):
        rand = random.random()
        for word in self.wordsCounter.keys():
            rand -= self.BiwordsCounter[word + " " + start] / self.sentenceMarkers
            if rand <= 0.0:
                return word


    # Given a sentence (sen), return the probability of
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")

        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity
    #(normalized inverse log probability)
    # N is the total words counter
    # perpelxity (W)= exp(-1/N* sum(log P(wi|wi-1)))
    def getCorpusPerplexity(self, corpus):
        words=[word for sen in corpus for word in sen[1:]]

        log_sum=0.0
        for word in words:
            p=self.prob(word)
            if p!=0:
                log_sum+=math.log(self.prob(word))
        return math.exp(-log_sum/len(words))

    def getCorpusPerlexity_sm(self,corpus):
        perplexity=0.0
        for sen in corpus:
            for word in sen:
                if word==start:
                    continue
                perplexity+=math.log(self.sm_prob(word))
        perplexity=-1/self.wordsTotal*perplexity
        return math.exp(perplexity)


    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        generateCorpus=[]
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            generateCorpus.append(sen)
            stringGenerated = str(prob) + " " + " ".join(sen)+"\n"
            print(stringGenerated, file=filePointer)
        return generateCorpus


	#endfor
    #enddef
#endclass


# Unigram language model
class UnigramModel(LanguageModel):
    def generateSentence(self):
        sent=[start]
        word=self.draw()
        while word!=end:
            sent.append(word)
            word=self.draw()
        sent.append(end)
        return sent

    def getSentenceProbability(self, sen):
        raw_prob=0.0
        for word in sen:
            if word==start:
                continue
            if self.wordsCounter[word]==0:
                return 0
            raw_prob= raw_prob + math.log(self.wordsCounter[word])-math.log(self.wordsTotal)
        return math.exp(raw_prob)
    #endddef
#endclass


#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def generateSentence(self):
        sentence=[]
        word=start
        while word!=end:
            sentence.append(word)
            word=self.sm_draw()
        sentence.append(end)
        return sentence

    def getSentenceProbability(self, sen):
        raw_prob = 0.0
        for word in sen:
            if word == start:
                continue
            if self.wordsCounter[word] == 0:
                return 0
            raw_prob = raw_prob + math.log(self.wordsCounter[word]+1) - math.log(self.wordsTotal+len(self.wordsCounter))
        return math.exp(raw_prob)
    #endddef
#endclass


# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def generateSentence(self):
        sent = []
        sent.append(start)
        cur = self.drawBi_start()

        while cur != end:
            sent.append(cur)
            cur = self.drawBi(cur)

        sent.append(end)
        return sent

    def getSentenceProbability(self, sen):
        prior=" "
        raw_prob=0.0
        for word in sen:
            if prior==start:
                raw_prob+=math.log(self.BiwordsCounter[word+" "+prior])-\
                math.log(self.sentenceMarkers)
            elif word !=start:
                raw_prob+=math.log(self.BiwordsCounter[word+" "+prior])-math.log(self.wordsCounter[prior])
            prior=word
        print(math.exp(raw_prob))
        return math.exp(raw_prob)

    def getCorpusPerplexity(self, corpus):
        perplexity=0.0

        prior=" "

        for sen in corpus:
            for word in sen:
                if word==start:
                    prior=word
                    continue
                if self.BiwordsCounter[word+" "+prior]==0.0:
                    # return Integer.MIN_VALUE
                    #  -inf not greater than 1e+20
                    return float("inf")
                if prior==start:
                    # new sentence
                    perplexity+=math.log(self.BiwordsCounter[word+" "+prior])-math.log(self.sentenceMarkers)
                    continue
                perplexity+=math.log(self.biprob(word, prior))
                prior=word
        perplexity=math.exp(-1/self.wordsTotal*perplexity)
        print(perplexity)

        return perplexity


    #endddef
#endclass

# Sample class for a unsmoothed unigram probability distribution
# Note:
#       Feel free to use/re-use/modify this class as necessary for your
#       own code (e.g. converting to log probabilities after training).
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        return self.counts[word]/self.total
    #enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word

	    #endif
	#endfor
    #enddef
#endclass

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)

    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')

    vocab = set()
    # Please write the code to create the vocab over here before the function preprocessTest

    print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")


    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    # Run sample unigram dist code
    print("Start Unsmoothed Unigram Model:==================")
    unigramModel=UnigramModel(trainCorpus)
    uniGenerateCorpus=unigramModel.generateSentencesToFile(20, "unigram_output.txt")
    unigramModel.getCorpusPerplexity(posTestCorpus)
    unigramModel.getCorpusPerplexity(negTestCorpus)
    print("End Unsmoothed Unigram Model. ")

    # Run smooth unigram dist code
    print("Start smooth Unigram Model: ================")
    SmoothedUnigramModel=SmoothedUnigramModel(trainCorpus)
    smGenerateCorpus=SmoothedUnigramModel.generateSentencesToFile(20,"smooth_unigram_output.txt")
    SmoothedUnigramModel.getCorpusPerlexity_sm(posTestCorpus)
    SmoothedUnigramModel.getCorpusPerlexity_sm(negTestCorpus)
    print("End smooth Unigram Model.")

    # run unsmooth bigram dist code
    print("start unsmooth Bigram Model: ==================")
    Bigrammodel=BigramModel(trainCorpus)
    biGenerateCorpus=Bigrammodel.generateSentencesToFile(20,"bigram_output.txt")
    Bigrammodel.getCorpusPerplexity(posTestCorpus)
    Bigrammodel.getCorpusPerplexity(negTestCorpus)
    print("end unsmooth Bigram Model.")



