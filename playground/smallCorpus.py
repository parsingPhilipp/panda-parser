#obtains a corpus of the size of the Polish dev-corpus from the German dev-corpus
#needs to be run from ../res/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/pred/conll/dev directory




def main():
    sentenceCounter = 0
    big = open('dev.German.pred.conll', 'r')
    small = open('dev.German_small.pred.conll', 'w')

    while sentenceCounter < 821:
        line = big.readline()
        small.write(line)
        if line == '\n':
            sentenceCounter += 1


    big.close()
    small.close()





if __name__ == '__main__':
    main()
