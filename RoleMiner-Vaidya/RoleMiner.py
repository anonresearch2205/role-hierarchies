import datetime
import os
import sys

import pandas as pd
import numpy as np
from termcolor import colored

prefix_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{prefix_dir}/..')
print(sys.path)

from readup import readup

# https://github.com/jbonifield3/RoleMiner/blob/master/RoleMiner/functions.py


def df_to_Boolean(df, RowCol, FactorCol):
    # Loads a csv with two columns into a
    # Used to Generate User-Permissions Matrix & User-Roles Matrix

    # If String - use as filepath
    if isinstance(df, str):
        df = pd.read_csv(df)

    factors = [RowCol, FactorCol]
    df = df[factors]
    df = df.drop_duplicates()
    df.fillna('Unknown', inplace=True)

    df = df.assign(value=1).set_index(factors)
    df = df.reindex(pd.MultiIndex.from_product(df.index.levels, names=df.index.names))

    df = (df.assign(value=df['value'].fillna(0).astype(int)).reset_index())
          # .groupby(level=0).apply(lambda x: x.ffill().bfill())
          # .reset_index())

    df = df.pivot(index=RowCol, columns=FactorCol, values='value')
    return df


def FastMiner(UP):
    # Remove Empty sets/no permission users
    UP = np.unique(np.array(UP), axis=0).tolist()
    UP = [x for x in UP if x != [0] * len(x)]

    # Initialize lists
    InitRoles = []
    GenRoles = []
    OrigCount = []
    GenCount = []
    Contributors = []

    for user in UP:

        if sum([user == x for x in InitRoles]) == 0:
            OrigCount.append(1)
            InitRoles.append(user)
        else:
            pos = [i for i, x in enumerate([user == x for x in InitRoles]) if x][0]
            OrigCount[pos] += 1

    InitRoles_iter = InitRoles

    for InitRole in InitRoles_iter:

        # Remove role from InitRoles
        # InitRoles_iter = [x for x in InitRoles if x != InitRole]

        for CandRole in InitRoles_iter:
            # New Role = InitRole ∩ CandRole (intersect Init Role with remaining roles)
            # NewRole = [x for x in CandRole if x == InitRole]

            NewRole = np.logical_and(list(map(bool, CandRole)), list(map(bool, InitRole))).astype(np.int64).tolist()

            if sum([NewRole == x for x in GenRoles]) == 0:
                GenCount.append(
                    OrigCount[[i for i, x in enumerate([InitRole == x for x in InitRoles]) if x][0]]
                    + OrigCount[[i for i, x in enumerate([CandRole == x for x in InitRoles]) if x][0]]
                )

                Contributors.append([CandRole, InitRole])
                GenRoles.append(NewRole)

    return np.array(GenRoles)


def Basic_RMP(UP, CandRoles, MaxRoles=100):
    cols = UP.columns
    UP = np.array(UP)
    Constraints = np.ones([UP.shape[0], CandRoles.shape[0]], dtype=int)
    OptRoles = []
    iters = 0
    UP_Remain = UP.copy()
    EntitlementCount = []

    for i in range(CandRoles.shape[0]):
        for j in range(UP_Remain.shape[0]):
            # Boolean logic: We check to see if Candidate Role is fully in the User
            # This can be modeled by all elements of the subtraction being less than 1
            Constraints[j, i] = (CandRoles[i, :] - UP_Remain[j, :] < 1).all()

    while len(OptRoles) < MaxRoles and iters < MaxRoles and np.sum(UP_Remain) > 0:
        # print(np.sum(UP_Remain))
        # Determine the Constraints that can be set to zero
        # TODO: Rewrite more Pythonic using list comp
        #       Update Basic Key algorithm to not recalc 0 for columns removed
        iters += 1

        # Calculate Basic Keys
        BasicKeys = []
        i = 0
        for Keys in Constraints.T:
            count = 0

            for C in range(UP_Remain.shape[0]):
                count += Keys[C] * np.sum(np.logical_and(CandRoles[i, :], UP_Remain[C, :])) * (
                            CandRoles[i, :] - UP[C, :] < 1).all()
            BasicKeys.append(count)
            i += 1

        # Add Role with highest Basic Key to OptRoles
        # Break ties by picking with lowest index
        BestRole = CandRoles[np.argmax(BasicKeys)]
        OptRoles.append(BestRole)

        # Delete Constraints from User Permissions
        for i in range(UP_Remain.shape[0]):
            if (UP[i, :] - BestRole > -1).all():
                UP_Remain[i, :] = UP_Remain[i, :] - BestRole

                # Map negative numbers to zero (these represent roles captured more than once, which is ok)
                UP_Remain[i, :][UP_Remain[i, :] < 0] = 0

        # Set Constraints in Best Role to zero
        Constraints[:, np.argmax(BasicKeys)] = 0
        EntitlementCount.append(sum(BestRole))

    # return New Rules as DF
    OptRoles = pd.DataFrame(data=OptRoles)

    return OptRoles, EntitlementCount


def get_UR(UP_matrix, roles_perms):
    UP_array = UP_matrix.to_numpy()
    RP_array = roles_perms.to_numpy()  # Transpose for multiplication

    UR_array = np.zeros((UP_array.shape[0], RP_array.shape[0]))
    for i in range(UP_array.shape[0]):
        UP_remain = UP_array[i, :]
        RP = RP_array.copy()

        for j in range(RP_array.shape[0]):
            if np.all(UP_remain == 0):
                continue
            if np.all(UP_remain >= 0):
                UP_remain_new = UP_remain - RP[j, :]
                if np.all(UP_remain_new >= 0):
                    UR_array[i, j] = 1
                    UP_remain = UP_remain_new

                    # update RP to remove the permissions of this role from all roles
                    RP = RP - RP[j, :]
                    RP[RP == -1] = 0

    return pd.DataFrame(UR_array.astype(int))


def check_UR_RP(UP: pd.DataFrame, UR: pd.DataFrame, RP: pd.DataFrame):
    UP_prime = UR.dot(RP)
    UP_prime[UP_prime >= 1] = 1
    return (UP_prime - UP == 0).all().all()


def main():
    start_time = datetime.datetime.now()
    print('Start time:', start_time)
    sys.stdout.flush()

    if len(sys.argv) != 2:
        print('Usage: ', end='')
        print(sys.argv[0], end=' ')
        print('<input-file>')
        return

    upfilename = sys.argv[1]
    up = readup(upfilename)

    max_roles = 1000
    print("Loading data and converting to Boolean matrix")

    # up to dataframe
    up_df = pd.DataFrame(columns=['user', 'permission'])
    for u in up:
        for p in up[u]:
            up_df = pd.concat([up_df, pd.DataFrame([{'user': u, 'permission': p}])])

    UP_matrix = df_to_Boolean(up_df, 'user', 'permission')

    print("Generating candidate roles using FastMiner")
    candidate_roles = FastMiner(UP_matrix)
    candidate_RP_df = pd.DataFrame(candidate_roles)
    UR = get_UR(UP_matrix, candidate_RP_df)
    valid = check_UR_RP(UP_matrix, UR, candidate_RP_df)
    if valid:
        print(colored("Initial candidate roles result in VALID UR * RP == UP", "white", "on_green"))
    else:
        print(colored("Initial candidate roles result in INVALID UR * RP == UP", "white", "on_red"))

    print("Optimizing roles using Basic_RMP")
    RP_df, EntitlementCount = Basic_RMP(UP_matrix, candidate_roles, max_roles)

    print("Role mining completed.")
    print(f"Total optimized roles: {len(RP_df)}")
    print(f"Entitlement counts: {EntitlementCount}")

    UR_df = get_UR(UP_matrix, RP_df)

    valid = check_UR_RP(UP_matrix, UR_df, RP_df)
    if valid:
        print(colored("Roles result in VALID UR * RP == UP", "white", "on_green"))
    else:
        print(colored("Roles result in INVALID UR * RP == UP", "white", "on_red"))

    num_edges = (UR_df == 1).sum().sum() + (RP_df == 1).sum().sum()
    print('Number of roles: ', RP_df.shape[0])
    print('Number of edges: ', num_edges)

    end_time = datetime.datetime.now()
    print('Time taken:', (end_time - start_time).total_seconds())

if __name__ == "__main__":
    main()


