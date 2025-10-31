#! /usr/bin/python3

import sys
import datetime
from minedgerolemining.readup import readup
from minedgerolemining.removedominators import readem
from minedgerolemining.removedominators import dmfromem
from minedgerolemining.algorithms.createILP import get_nroles
from minedgerolemining.algorithms.createILP import run_ilp_and_get_roles

import gurobipy as gp
from itertools import combinations


def main():
    print('Start time:', datetime.datetime.now())
    sys.stdout.flush()

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print('Usage: ', end='')
        print(sys.argv[0], end=' ')
        print('<input-up-file> <input-em-file>')
        return

    up = readup(sys.argv[1])
    if not up:
        print('Empty up; nothing to do.')
        return

    em = readem(sys.argv[2])
    if not em:
        print('Empty em; nothing to do.')
        return

    dm = dmfromem(em)

    """
    print('em:')
    for e in em:
        print(str(e)+': '+str(em[e]))
    print('dm:')
    for d in dm:
        print(str(d)+': '+str(dm[d]))
    """

    if tuple((-1, -1)) not in dm:
        print("Hm...something wrong; no (-1,-1) in dm. Quitting...")
        return

    # First setup the roles
    roletoedgemap = dict()
    edgetorolemap = dict()
    rolenum = 0
    for e in dm[tuple((-1, -1))]:
        roletoedgemap[rolenum] = e
        edgetorolemap[e] = rolenum
        rolenum += 1

    print('# roles:', rolenum)
    sys.stdout.flush()

    # Now users & perms to those roles
    usertorolesmap = dict()
    roletousersmap = dict()
    permtorolesmap = dict()
    roletopermsmap = dict()

    for r in roletoedgemap:
        roletousersmap[r] = set()
        roletopermsmap[r] = set()

        # A simple kind of BFS
        q = list()
        q.append(roletoedgemap[r])

        while q:
            e = q.pop(0)
            u = e[0]
            p = e[1]

            (roletousersmap[r]).add(u)
            (roletopermsmap[r]).add(p)

            if u not in usertorolesmap:
                usertorolesmap[u] = set()
            (usertorolesmap[u]).add(r)

            if p not in permtorolesmap:
                permtorolesmap[p] = set()
            (permtorolesmap[p]).add(r)

            if e in dm:
                q.extend(dm[e])

    """
    print('roletousersmap:')
    for r in roletousersmap:
        print(str(r)+': '+str(roletousersmap[r]))

    print('roletopermsmap:')
    for r in roletopermsmap:
        print(str(r)+': '+str(roletopermsmap[r]))

    print('usertorolesmap:')
    for u in usertorolesmap:
        print(str(u)+': '+str(usertorolesmap[u]))

    print('permtorolesmap:')
    for p in permtorolesmap:
        print(str(p)+': '+str(permtorolesmap[p]))
    """

    rset = set(roletoedgemap.keys())
    k = 2  # for each k-sized subset of roles

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    # randfilename = "/tmp/randfile-"+str(random.getrandbits(32))+".txt"
    # print('randfilename:', randfilename)

    for rsubtuple in combinations(rset, k):
        rsubset = set(rsubtuple)
        print('rsubset:', rsubset)
        sys.stdout.flush()

        # First, the num edges in this subset of roles
        currnedges = 0
        for r in rsubset:
            currnedges += len(roletousersmap[r])
            currnedges += len(roletopermsmap[r])

        thisup = dict()
        thisusers = set()
        thisperms = set()
        thisnedges = 0

        for r in rsubset:
            uset = roletousersmap[r]
            pset = roletopermsmap[r]
            thisusers.update(uset)
            thisperms.update(pset)
            for u in uset:
                if u not in thisup:
                    thisup[u] = set()
                (thisup[u]).update(pset)
                thisnedges += len(pset)

        # dumpup(thisup, randfilename)
        # thisup = readup(randfilename)
        # print('thisup:', thisup)

        thismaxnroles = get_nroles(thisup, thisusers, thisperms, thisnedges)
        # not sure that this is tight
        m = gp.Model("minedgesIQP", env=env)
        m.Params.TimeLimit = 30  # 180 seconds only, per run
        num_edges, roles_mapped, roles_as_edges = run_ilp_and_get_roles(thisup, thisusers, thisperms, thismaxnroles, m)
        if num_edges < currnedges:
            print('currnedges:', currnedges, ', num_edges:', num_edges, ', num_new_roles:', len(roles_mapped.keys()))
            print('curr_roles + {users}, {perms}:')
            for r in rsubset:
                print('\t' + str(r) + ': ' + str(sorted(roletousersmap[r])) + ', ' + str(sorted(roletopermsmap[r])))
            print('roles_mapped:')
            for r in roles_mapped:
                print('\t' + str(r) + ': ', end='')
                userset = set()
                permset = set()
                for e in roles_mapped[r]:
                    if e[0] == 'u':
                        userset.add(int(e[2:]))
                    else:
                        permset.add(int(e[2:]))
                print(str(sorted(userset)) + ', ' + str(sorted(permset)))

            sys.stdout.flush()

    print('End time:', datetime.datetime.now())
    sys.stdout.flush()


if __name__ == '__main__':
    main()