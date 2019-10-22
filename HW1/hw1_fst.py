from fst import *

# here are some predefined character sets that might come in handy.
# you can define your own
AZ = set("abcdefghijklmnopqrstuvwxyz")
VOWS = set("aeiou")
CONS = set("bcdfghjklmnprstvwxz")
E = set("e")
DOUBLE=set("ntpr")
DOUBLE_e=set("tp")
I=set("i")
AUO=set("auo")



# Implement your solution here
def buildFST():
    print("Your task is to implement a better FST in the buildFST() function, using the methods described here")
    print("You may define additional methods in this module (hw1_fst.py) as desired")
    #
    # The states (you need to add more)
    # ---------------------------------------
    #
    f = FST("q0") # q0 is the initial (non-accepting) state
    f.addState("q1") # a non-accepting state
    f.addState("q_ing") # a non-accepting state
    f.addState("q_EOW", True) # an accepting state (you shouldn't need any additional accepting states)


    #
    # The transitions (you need to add more):
    # ---------------------------------------
    # transduce every element in this set to itself:
    f.addSetTransition("q0", AZ, "q1")
    # AZ-E =  the set AZ without the elements in the set E
    # f.addSetTransition("q1", AZ-E, "q1")

    f.addState("q1.1")
    f.addSetTransition("q0",VOWS,"q1.1")
    # get rid of this transition! (it overgenerates):
    f.addSetTransition("q1", AZ-VOWS, "q1")

    # rule 1: end with e, remove e and add -ing
    f.addState("q_e")
    f.addTransition("q1","e","","q_e")
    f.addTransition("q_e","","ing","q_EOW")

    f.addState("q_double")
    f.addState("q_not_double")

    # rule 2: end with ntpr, double +ing
    # combined e with t,p; er, en not double:
    # e + a/e/i/o/u
    for i in range(0,26):
        if chr(i+97) in DOUBLE_e:
            # double the last char
            f.addTransition("q_e",chr(i+97),"e"+chr(i+97)+chr(i+97),"q_double")
            # no double char
            f.addTransition("q_e",chr(i+97),"e"+chr(i+97),"q_not_double")
        else:
            # return to the q1

            f.addTransition("q_e",chr(i+97),"e"+chr(i+97),"q1")

    f.addTransition("q_double", "", "ing", "q_EOW")
    f.addSetTransition("q_not_double", AZ - VOWS, "q1")


    # following with e, recursively
    # rule 3: following with i, new state q_i
    # rule 4: following with a,u,o, new state q_auo
    f.addState("q_i")
    f.addState("q_auo")
    f.addTransition("q_not_double","i","","q_i")
    f.addTransition("q_not_double","e","","q_e")
    f.addSetTransition("q_not_double",VOWS-I-E,"q_auo")


    # rule 3:
    f.addTransition("q1","i","","q_i")
    # q_i, following e, new state
    f.addState("q_ie")
    f.addTransition("q_i","e","","q_ie")
    # end with ie
    f.addTransition("q_ie","","ying","q_EOW")

    # q_i, not following e
    # double test
    for i in range(0,26):
        if i!=4:
           if chr(i+97) in DOUBLE:
               f.addTransition("q_i",chr(i+97),"i"+chr(i+97)+chr(i+97),"q_double")
               f.addTransition("q_i",chr(i+97),"i"+chr(i+97),"q_not_double")
           else:
               f.addTransition("q_i",chr(i+97),"i"+chr(i+97),"q1")



    # not end with ie, return ie.
    for i in range(0,26):
        f.addTransition("q_ie",chr(i+97),"ie"+chr(i+97),"q1")
    f.addSetTransition("q_ie",AZ,"q1")

    # rule 4:end with a,u,o
    f.addState("q1.2")
    f.addSetTransition("q0",AZ-VOWS,"q1.2")
    f.addState("q_CONS_auo")
    f.addSetTransition("q1.2",VOWS-E-I,"q_CONS_auo")
    f.addState("q_vow_auo")
    f.addSetTransition("q1.1",VOWS-E-I,"q_vow_auo")





    # not following e
    for i in range(0,26):
        if chr(i+97) in DOUBLE:
            f.addTransition("q_CONS_auo",chr(i+97),chr(i+97)+chr(i+97),"q_double")
            f.addSetTransition("q_CONS_auo",chr(i+97),"q_not_double")
            f.addSetTransition("q_vow_auo",chr(i+97),"q_not_double")
        else:
            f.addSetTransition("q_CONS_auo",chr(i+97),"q1")
            f.addSetTransition("q_vow_auo",chr(i+97),"q1")


    # map the empty string to ing:
    f.addTransition("q1", "", "ing", "q_EOW")

    # Return your completed FST
    return f


if __name__ == "__main__":
    # Pass in the input file as an argument
    if len(sys.argv) < 2:
        print("This script must be given the name of a file containing verbs as an argument")
        quit()
    else:
        file = sys.argv[1]
    #endif

    # Construct an FST for translating verb forms 
    # (Currently constructs a rudimentary, buggy FST; your task is to implement a better one.
    f = buildFST()
    # Print out the FST translations of the input file
    f.parseInputFile(file)
