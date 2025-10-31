#! /usr/bin/python3

import sys
import time
import os
import datetime

from readup import readup
from readup import uptopu
from removedominatorsbp import removedominators
from removedominatorsbp import saveem
from removedominatorsbp import readem
from findcliquesbp import find_bicliquesbp
from maxsetsbp import run


def printem(em):
    for e in em:
        print('\t' + str(e) + ': ' + str(em[e]))
    sys.stdout.flush()


def addtoem(bclist, em, seq):
    for b in bclist:  # b is a biclique, i.e., list of edges
        firstedge = None
        for e in b:  # e is an edge
            if not firstedge:
                em[e] = tuple((-1, -1, seq))
                firstedge = e
            else:
                em[e] = tuple((firstedge[0], firstedge[1], seq))
            seq += 1

    return seq


def addtobclist(c, bclist, THRESHOLD):
    s = set(c)
    for b in bclist:
        s.difference_update(b)

    # print('s (after difference_updates):', s)

    if len(s) >= THRESHOLD:
        l = list(s)
        bclist.append(l)
        return l

    return None


def run_largebicliques(upfilename, bcsize_THRESHOLD, nbc_THRESHOLD):
    up = readup(upfilename)
    if not up:
        return

    pu = uptopu(up)

    nedges = 0
    for u in up:
        nedges += len(up[u])

    # em
    seq = 0
    em = dict()
    dm = dict()
    emfilename = upfilename + '-em.txt'

    # if os.path.exists(emfilename):
    #    print('Reading em from', emfilename, '...', end='')
    #    sys.stdout.flush()
    #    em = readem(emfilename)
    #    print('done!')
    #    sys.stdout.flush()
    # else:
    #    em = dict()

    timeone = time.time()
    if not os.path.isfile(emfilename):
        print('Removing doms + zero-neighbour edges...')
        sys.stdout.flush()
        seq = removedominators(em, dm, up, seq)
        timetwo = time.time()
        print('done! Time taken:', timetwo - timeone)
        sys.stdout.flush()
        print('Saving em to', emfilename, end=' ')
        sys.stdout.flush()
        saveem(em, emfilename)
        print('done!')
        sys.stdout.flush()
    else:
        print('Reading em from', emfilename, end=' ')
        sys.stdout.flush()
        em = readem(emfilename)
        print('done!')
        sys.stdout.flush()
        print('Determining seq', end=' ')
        sys.stdout.flush()
        for e in em:
            if seq <= em[e][2]:
                seq = em[e][2] + 1
        print('done!')
        sys.stdout.flush()

    nbc_COUNT = 0
    nc = 0
    bclist = list()  # list of bicliques we identify
    timeone = time.time()
    start_time = time.time()

    # print('em before find_bicliques:')
    # printem(em)

    bicliquesexhausted = False
    toomanycliques_THRESHOLD = 3000000

    while nbc_COUNT < nbc_THRESHOLD and not bicliquesexhausted:

        print('(Re)starting biclique enumeration')
        sys.stdout.flush()

        nc = 0
        largebicliquefound = False
        for c in find_bicliquesbp(em, up, pu, list()):
            # if not nc:
            #    timetwo = time.time()
            #    print('First clique found, time:', timetwo - timeone, '...')
            #    sys.stdout.flush()
            nc += 1

            if nc >= toomanycliques_THRESHOLD:
                # reduce the nbc_THRESHOLD and try again
                bcsize_THRESHOLD -= 50  # reduce by 50 each time
                print('toomanycliques_THRESHOLD,', toomanycliques_THRESHOLD, 'exceeded. Reducing bcsize_THRESHOLD to',
                      bcsize_THRESHOLD)
                sys.stdout.flush()
                largebicliquefound = True  # to bypass check below for whether
                # bicliques are exhausted
                break

            if not nc % 10000:
                timetwo = time.time()
                print('# bicliques:', nc, '; time:', timetwo - timeone, '...')
                sys.stdout.flush()
                timeone = time.time()

            if len(c) >= bcsize_THRESHOLD:
                print('large biclique found:', c)
                print('large biclique size:', len(c))
                sys.stdout.flush()
                largebicliquefound = True
                seq = addtoem([c], em, seq)
                nbc_COUNT += 1
                break

        if not largebicliquefound:
            bicliquesexhausted = True

    print('nbc_COUNT:', nbc_COUNT, ', bicliquesexhausted:', bicliquesexhausted)
    end_time = time.time()
    print('Time taken to find large bicliques:', end_time-start_time)
    # print('em after find_bicliques:')
    # printem(em)

    print('Saving em to', emfilename, end='...')
    sys.stdout.flush()
    saveem(em, emfilename)
    if len(em) < nedges:
        print('Some edges remain, invoking maxsetsbp')
        sys.stdout.flush()

        objval, roles = run(upfilename)
        # print('em after maxsetsbp:')
        # printem(em)
        return objval, roles
    else:
        print('No edges remain, no need to invoke maxsetsbp')
        sys.stdout.flush()


    print('done!')

    print('End time:', datetime.datetime.now())
    sys.stdout.flush()
    return 0, []


def main():
    print('Start time:', datetime.datetime.now())
    sys.stdout.flush()

    if len(sys.argv) != 4:
        print('Usage: ', end='')
        print(sys.argv[0], end=' ')
        print('<input-file> <biclique-size-threshold> <num-bicliques>')
        return

    # Arguments:
    # (1) input up file
    # (2) at what size we consider a biclique sufficiently large
    # (3) after removing how many of those large bicliques we invoke maxsetsbp

    upfilename = sys.argv[1]
    bcsize_THRESHOLD = int(sys.argv[2])
    nbc_THRESHOLD = int(sys.argv[3])


    run_largebicliques(upfilename, bcsize_THRESHOLD, nbc_THRESHOLD)


if __name__ == '__main__':
    main()
