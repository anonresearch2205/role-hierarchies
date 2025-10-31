import datetime
import os
import sys

import numpy as np
import pandas as pd
from termcolor import colored

prefix_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{prefix_dir}/..')
print(sys.path)

from readup import readup
from RoleMiner import df_to_Boolean, FastMiner, get_UR, check_UR_RP
from utils import (calculate_number_of_edges_in_rbac, check_roles, make_biclique, is_biclique, get_roles_mapped,
                   inverse_map)


def get_unique_permission_sets(up: dict) -> dict:
    perms_sets_dict = dict()
    for u in up:
        perms = tuple(up[u])
        if perms in perms_sets_dict:
            perms_sets_dict[perms] += 1
        else:
            perms_sets_dict[perms] = 1
    return perms_sets_dict


def create_constraint_matrix(up_df):
    # df_to_Boolean()
    pass


def augment_up(up: dict) -> dict:
    up_augmented = dict()

    for u in up:
        up_augmented[f'u{u}'] = [f'p{p}' for p in up[u]]
    return up_augmented


def compute_basic_keys(constraints: np.ndarray, up_bool: np.ndarray,
                       up_remain: np.ndarray, candidate_roles: np.ndarray):
    # Calculate Basic Keys
    basic_keys = []
    i = 0
    for constr in constraints.T:
        count = 0

        # Basic Key: For each column of matrix {c_ij}, which represents the users assigned to a role j,
        # count the number of constraints {Sum_j c_ih * r_jt = 1 for x_it = 1} being satisfied by letting the cells
        # of such column to be 1 except the cells predetermined in step 2 to be 0.

        # In other words: The number of constraints satisfied when all users (except the ones already marked 0) are
        # assigned to a role.
        for C in range(up_remain.shape[0]):
            # constr[C] -> roles assigned to user C
            # np.logical_and(candidate_roles[i, :], up_remain[C, :]) -> permissions satisfied with this role
            # (candidate_roles[i, :] - up_bool[C, :] < 1) ->
            count += (constr[C] * np.sum(np.logical_and(candidate_roles[i, :], up_remain[C, :]))
                      * (candidate_roles[i, :] - up_bool[C, :] < 1).all())
        basic_keys.append(count)
        i += 1
    return basic_keys


def compute_edge_keys(constraints: np.ndarray, up_bool_orig: np.ndarray, up_bool: np.ndarray,
                      up_remain: np.ndarray, candidate_roles: np.ndarray):

    # Calculate edge keys
    edge_keys = []
    i = 0
    for constr_col in constraints.T:
        count = 0

        UP = pd.DataFrame(up_bool_orig).astype(int)
        UP_trimmed = pd.DataFrame(up_bool).astype(int)
        # compute counts of each unique permission set
        unique_rows, counts = np.unique(UP.values, axis=0, return_counts=True)
        UP_counts = pd.DataFrame(unique_rows)
        UP_counts["count"] = counts
        # UP_counts = UP.groupby(UP.columns.tolist()).size().reset_index(name="count")
        # multiply each row with its count
        # we use this later to compute edge key
        UP_counted = UP_trimmed.iloc[:, :].mul(UP_counts["count"], axis=0)



        # Edge Key: For each column of matrix {c_ij}, which represents the users assigned to a role j,
        # Determine the constraints {Sum_j c_ih * r_jt = 1 for x_it = 1} satisfied by letting the
        # cells of such column to be 1 except the cells predetermined in step 2 to be 0.
        # For each constraint satisfied, there is a count of the unique permission set (which is the number of
        # permissions in this set). Sum of these counts for a constraint is the edge key for that constraint

        # In other words: The number of constraints satisfied when all users (except the ones already marked 0) are
        # assigned to a role.
        for C in range(up_remain.shape[0]):
            # check that this role i does not add extra permission than the ones defined in the UP for this user C
            # if (candidate_roles[i, :] - up_bool[C, :] < 1).all():
            # get the permission indices where the permissions of the role i match permissions of user in UP
            matching_perms_indices = UP_trimmed.T.index[np.logical_and(candidate_roles[i, :], up_bool[C, :])].tolist()
            val = 0
            # for these permission indices, get the "associated count"
            # (number of times this permission set was repeated in the UP)
            val += (constr_col[C]
                    * np.sum(UP_trimmed.iloc[C, matching_perms_indices]
                           * UP_counted.iloc[C, matching_perms_indices])
                    * (candidate_roles[i, :] - up_bool[C, :] < 1).all())

            if val >= 0:
                count += val

        edge_keys.append(count)
        i += 1
    return edge_keys


def pick_edge_key_index(edge_keys: np.array, candidate_roles: np.ndarray):
    max_edge_key_value = np.max(edge_keys)
    max_indices = np.argwhere(edge_keys == max_edge_key_value).flatten()
    fewest_permissions = candidate_roles.shape[1]
    role_with_fewest_perms = max_indices[0]
    for i in max_indices:
        num_perms = np.sum(candidate_roles[i, :])
        if num_perms < fewest_permissions:
            fewest_permissions = num_perms
            role_with_fewest_perms = i
    return role_with_fewest_perms




def edgeRMPGreedy(up: dict, max_roles=100):
    # Step 1: Get n unique user permission sets. Count the number of the repetitions of every unique permission set,
    # {u_1, u_2 ... u_n}.
    # perms_sets_dict: dictionary to keep track of unique permission sets and keep track of counts (repetitions)
    perms_sets_dict = get_unique_permission_sets(up)
    print(perms_sets_dict)
    up_trimmed = up

    # up to dataframe
    up_df = pd.DataFrame(columns=['user', 'permission'])
    for u in up_trimmed:
        for p in up_trimmed[u]:
            up_df = pd.concat([up_df, pd.DataFrame([{'user': u, 'permission': p}])])

    # get UP as a boolean matrix
    print("Loading data and converting to Boolean matrix")
    up_bool_orig = df_to_Boolean(up_df, 'user', 'permission')
    UP = pd.DataFrame(up_bool_orig).astype(int)

    up_bool = np.unique(up_bool_orig, axis=0)
    # get candidate roles from FastMiner
    print("Generating candidate roles using FastMiner")
    candidate_roles = FastMiner(up_bool)

    candidate_RP_df = pd.DataFrame(candidate_roles)
    UR = get_UR(UP, candidate_RP_df)
    valid = check_UR_RP(UP, UR, candidate_RP_df)
    if valid:
        print(colored("Initial candidate roles result in VALID UR * RP == UP", "white", "on_green"))
    else:
        print(colored("Initial candidate roles result in INVALID UR * RP == UP", "white", "on_red"))

    num_edges = (UR == 1).sum().sum() + (candidate_RP_df == 1).sum().sum()

    print('Number of roles in candidate RBAC policy: ', candidate_RP_df.shape[0])
    print('Number of edges in candidate RBAC policy: ', num_edges)


    constraints = np.ones([up_bool.shape[0], candidate_roles.shape[0]], dtype=int)

    up_bool = np.array(up_bool)
    up_remain = up_bool.copy()

    # Step 2: Determine some variables c_ij to 0, according to Sum_jq c_ij * r_jt = 0.
    for i in range(candidate_roles.shape[0]):
        for j in range(up_remain.shape[0]):
            # Boolean logic: We check to see if Candidate Role is fully in the User
            # This can be modeled by all elements of the subtraction being less than 1
            constraints[j, i] = (candidate_roles[i, :] - up_remain[j, :] < 1).all()

    opt_roles = []
    iters = 0
    while len(opt_roles) < max_roles and iters < max_roles and np.sum(up_remain) > 0:
        iters += 1
        # up_bool = up_remain.copy()
        edge_keys = compute_edge_keys(constraints, up_bool_orig, up_bool, up_remain, candidate_roles)

        # Add Role with highest Edge Key to OptRoles Break ties by picking with lowest index
        # if np.all(np.array(edge_keys) == 0):
        #     continue
        # print('Edge keys: ', edge_keys)
        best_edge_key_index = pick_edge_key_index(edge_keys, candidate_roles)

        best_role = candidate_roles[best_edge_key_index]
        opt_roles.append(best_role)

        # Delete Constraints from User Permissions
        for i in range(up_remain.shape[0]):
            if (up_bool[i, :] - best_role > -1).all():
                up_remain[i, :] = up_remain[i, :] - best_role

                # Map negative numbers to zero (these represent roles captured more than once, which is ok)
                up_remain[i, :][up_remain[i, :] < 0] = 0

        # Set Constraints in Best Role to zero
        # constraints[:, np.argmax(edge_keys)] = 0
        constraints[:, best_edge_key_index] = 0

    RP = pd.DataFrame(data=opt_roles)
    UR = get_UR(UP, RP)

    return UP, UR, RP


def main():
    start_time = datetime.datetime.now()
    print('Start time:', start_time)
    sys.stdout.flush()

    if len(sys.argv) != 3:
        print('Usage: ', end='')
        print(sys.argv[0], end=' ')
        print('<input-file> <max-roles>')
        return

    upfilename = sys.argv[1]
    max_roles = int(sys.argv[2])
    up = readup(upfilename)

    UP, UR, RP = edgeRMPGreedy(up, max_roles=max_roles)
    valid = check_UR_RP(UP, UR, RP)
    if valid:
        print(colored("Roles result in VALID UR * RP == UP", "white", "on_green"))
    else:
        print(colored("Roles result in INVALID UR * RP == UP", "white", "on_red"))

    num_edges = (UR == 1).sum().sum() + (RP == 1).sum().sum()

    print('Number of roles: ', RP.shape[0])
    print('Number of edges: ', num_edges)
    end_time = datetime.datetime.now()
    print('Time taken:', (end_time - start_time).total_seconds())


if __name__ == '__main__':
    main()
