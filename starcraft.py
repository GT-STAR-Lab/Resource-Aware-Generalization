# Storing the List of Species with their traits in a dataframe

import pandas as pd

team_data = {
    'species': ['Zealots', 'Stalkers', 'Marines', 'Marauders'],
    'armor': [1, 1, 0, 1],
    'hitPoints': [100, 80, 45, 125],
    'plasmaShield': [50, 80, 0, 0],
    'gAttack': [8, 13, 6, 5],
    'aAttack': [0, 13, 6, 0],
    'gDPS': [18.6, 9.7, 9.8, 9.3],
    'aDPS': [0, 9.7, 9.8, 0],
    'cooldown': [0.86, 1.34, 0.61, 1.07],
    'speed': [3.15, 4.13, 3.15, 3.15],
    'range': [0, 6, 5, 6],
    'sight': [9, 10, 9, 10]
}

df_team = pd.DataFrame(team_data, columns=['species', 'armor', 'hitPoints', 'plasmaShield', 'gAttack', 'aAttack', 'gDPS', 'aDPS', 'cooldown', 'speed', 'range', 'sight'])
df_team.set_index(['species'], inplace=True)

"""
    StarCraft II Tests - 10 Teams with 5 teams having two species, 3 teams with 3 species and 2 teams with 4 species
    Each trial computes the assignments for our approach and the baselines
    All data across 10 trials are stored for further analysis
"""
import numpy as np
import time
import math
from docplex.mp.advmodel import Model
import cvxpy
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering
import random


def CreateRobotDistribution(Y_desired, Q_target, n_robot_per_species, task_restrict=None):
    num_tasks = len(Y_desired)
    num_strategies = Y_desired[0].shape[0]
    n_species = Q_target.shape[0]
    if task_restrict is None:
        task_restrict = range(num_tasks)
    X_target = np.zeros((num_tasks, n_species))
    """m = np.random.randint(0, num_tasks)
    while True:
        X_target[m, :] = np.random.choice(np.arange(sum(n_robot_per_species)), size=n_species)
        if sum(X_target.ravel(order='C')) == sum(n_robot_per_species):
            break"""
    for s in range(n_species):
        R = np.random.choice(task_restrict, size=n_robot_per_species[s])
        for m in task_restrict:
            X_target[m, s] = np.sum(R == m)

    Z_target = np.zeros([num_tasks, num_strategies])
    comp = np.dot(X_target, Q_target)
    error_min = np.zeros(num_tasks)
    error_exact = np.zeros(num_tasks)
    for i in range(num_tasks):
        find_min = [np.linalg.norm(comp[i, :] - Y_desired[i][0, :]), np.linalg.norm(comp[i, :] - Y_desired[i][1, :]), np.linalg.norm(comp[i, :] - Y_desired[i][2, :])]
        ind = find_min.index(min(find_min))
        Z_target[i, ind] = 1
        error_exact[i] = np.linalg.norm(comp[i, :] - Y_desired[i][ind, :]) / np.linalg.norm(Y_desired[i][ind, :])
        #if np.linalg.norm(comp[i, :]) <= np.linalg.norm(Y_desired[i][ind, :]):
        #    error_min[i] = np.linalg.norm(comp[i, :] - Y_desired[i][ind, :]) / np.linalg.norm(Y_desired[i][ind, :])
        for j in range(num_strategies):
            if comp[i, j] > Y_desired[i][ind, j]:
                error_min[i] += 0
            else:
                error_min[i] += (comp[i, j] - Y_desired[i][ind, j]) ** 2
        error_min[i] = math.sqrt(error_min[i]) / np.linalg.norm(Y_desired[i][ind, :])
    return error_min, error_exact, X_target.astype(np.int32), Z_target


def baseline_transfer(Y_desired, Q_target, n_target):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # X_target: assignment of each agent in the target team

    num_tasks = Y_desired.shape[0]
    n_target_species = Q_target.shape[0]

    X_sol = cvxpy.Variable((num_tasks, n_target_species), integer=True)

    # minimize trait mismatch
    mismatch_mat = Y_desired - cvxpy.matmul(X_sol, Q_target)  # trait mismatch matrix
    obj = cvxpy.Minimize(cvxpy.pnorm(mismatch_mat, 2))
    # obj = cvxpy.Minimize(cvxpy.sum(cvxpy.multiply(mismatch_mat, mismatch_mat)))

    # ensure each agent is only assigned to one task
    constraints = [cvxpy.matmul(X_sol.T, np.ones([num_tasks, 1])) <= np.array([n_target]).T, X_sol >= 0]

    # solve for X_target
    opt_prob = cvxpy.Problem(obj, constraints)
    opt_prob.solve(solver=cvxpy.CPLEX)
    X_target = X_sol.value

    return X_target


def global_baseline(Y_desired, Q_target, n_target):
    # Compute the centroid for each task
    # Do the task assignment - specieswise transfer

    num_strategies, num_traits = Y_desired[0].shape
    num_tasks = len(Y_desired)
    Y_baseline = np.zeros([num_tasks, num_traits])
    n_species = Q_target.shape[0]
    # n_target = n_agents_per_species * np.ones(n_species)

    for i in range(num_tasks):
        temp = np.zeros(num_traits)
        for j in range(num_strategies):
            temp += Y_desired[i][j, :]
        Y_baseline[i, :] = temp / num_strategies
    X_target = baseline_transfer(Y_baseline, Q_target, n_target)

    Z_target = np.zeros([num_tasks, num_strategies])
    comp = np.dot(X_target, Q_target)
    error_min = np.zeros(num_tasks)
    error_exact = np.zeros(num_tasks)
    for i in range(num_tasks):
        find_min = [np.linalg.norm(comp[i, :] - Y_desired[i][0, :]), np.linalg.norm(comp[i, :] - Y_desired[i][1, :]),
                    np.linalg.norm(comp[i, :] - Y_desired[i][2, :])]
        # find_min = [np.linalg.norm(comp[i, :] - Y_desired[i][0, :]), np.linalg.norm(comp[i, :] - Y_desired[i][1, :])]
        ind = find_min.index(min(find_min))
        Z_target[i, ind] = 1
        error_exact[i] = np.linalg.norm(comp[i, :] - Y_desired[i][ind, :]) / np.linalg.norm(Y_desired[i][ind, :])
        # if np.linalg.norm(comp[i, :]) <= np.linalg.norm(Y_desired[i][ind, :]):
        #    error_min[i] = np.linalg.norm(comp[i, :] - Y_desired[i][ind, :]) / np.linalg.norm(Y_desired[i][ind, :])
        for j in range(num_strategies):
            if comp[i, j] > Y_desired[i][ind, j]:
                error_min[i] += 0
            else:
                error_min[i] += (comp[i, j] - Y_desired[i][ind, j]) ** 2
        error_min[i] = math.sqrt(error_min[i]) / np.linalg.norm(Y_desired[i][ind, :])

    return error_min, error_exact, X_target, Z_target


def baseline(Y_desired, Q_target, n_target, obj_opt):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # Z_target: a matrix indicating which strategies were chosen for each task (n_tasks x n_strategies)
    # X_target: assignment of each agent in the target team
    # error_target: minimum trait mismatch error achieved
    # X_sol: solved cplex variable

    mdl = Model('Baseline')
    trait_mismatch_all = []
    # num_tasks = Y_desired[0].shape[0]
    num_tasks = len(Y_desired)
    n_target_species, n_traits = Q_target.shape
    # n_target = np.ones(n_target_species) * n_agents_per_species
    n_strategies = Y_desired[0].shape[0]

    X_target = np.zeros((num_tasks, n_target_species))
    Z_target = np.zeros((num_tasks, n_strategies))

    for i in range(num_tasks):
        k = np.random.randint(0, n_strategies)
        # k = 1
        for j in range(n_strategies):
            if j == k:
                Z_target[i, j] = 1

    X_sol = mdl.integer_var_list((num_tasks * n_target_species), name='x')

    [mdl.add_constraint(mdl.sum(X_sol[(nj * n_target_species) + i] for nj in range(0, num_tasks)) <= n_target[i]) for i
     in range(0, n_target_species)]

    # To ensure that the traits are at least equal to the desired
    for i in range(num_tasks):
        for j in range(n_strategies):
            for k in range(n_traits):
                mdl.add_constraint(Z_target[i, j] * Y_desired[i][j, k] <= mdl.dot(
                    X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]))

    # Print Constraints
    # for i in range(2 * n_strategies + 2 * num_tasks + 2 * n_traits):
    #    print(mdl.get_constraint_by_index(i))

    if obj_opt == "mismatch":

        error1 = 0.0
        error2 = 0.0
        total_error = []
        # minimize trait mismatch
        trait_mismatch = 0.0
        for i in range(num_tasks):
            for j in range(n_strategies):
                error2 = 0.0
                for k in range(n_traits):
                    error2 += ((Y_desired[i][j, k] ** 2) - (
                            2 * mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]) *
                            Y_desired[i][j, k]))  # trait mismatch matrix wrt Strategy i
                trait_mismatch_all.append(error2)

        for i in range(num_tasks):
            error1 = 0.0
            for j in range(n_traits):
                error1 += (mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[j, :]) ** 2)
            total_error.append(error1)

        # Have to split the error expression
        for i in range(num_tasks):
            for j in range(n_strategies):
                trait_mismatch += Z_target[i, j] * trait_mismatch_all[
                    i * n_strategies + j]  # only count if the strategy is used
            trait_mismatch += total_error[i]

        opt_prob = mdl.minimize(trait_mismatch)

    if obj_opt == "agents":

        total_team = 0
        for i in range(num_tasks * n_target_species):
            total_team += X_sol[i]
        opt_prob = mdl.minimize(total_team)

    # print(mdl.get_objective_expr())
    # print(mdl.print_information())

    # solve for X_target
    mdl.solve(url=None, key=None)  # , "Solve failed"
    # print("status:", mdl.solve_status)

    k = 0
    for i in range(num_tasks):
        for j in range(n_target_species):
            X_target[i, j] = X_sol[k].solution_value
            k += 1

    # mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def baseline_wc(Y_desired, Q_target, n_target, obj_opt):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # Z_target: a matrix indicating which strategies were chosen for each task (n_tasks x n_strategies)
    # X_target: assignment of each agent in the target team
    # error_target: minimum trait mismatch error achieved
    # X_sol: solved cplex variable

    mdl = Model('Baseline')
    trait_mismatch_all = []
    # num_tasks = Y_desired[0].shape[0]
    num_tasks = len(Y_desired)
    n_target_species, n_traits = Q_target.shape
    # n_target = np.ones(n_target_species) * n_agents_per_species
    n_strategies = Y_desired[0].shape[0]

    X_target = np.zeros((num_tasks, n_target_species))
    Z_target = np.zeros((num_tasks, n_strategies))

    for i in range(num_tasks):
        k = np.random.randint(0, n_strategies)
        for j in range(n_strategies):
            if j == k:
                Z_target[i, j] = 1

    X_sol = mdl.integer_var_list((num_tasks * n_target_species), name='x')

    [mdl.add_constraint(mdl.sum(X_sol[(nj * n_target_species) + i] for nj in range(0, num_tasks)) <= n_target[i]) for i
     in range(0, n_target_species)]

    # Print Constraints
    # for i in range(2 * n_strategies + 2 * num_tasks + 2 * n_traits):
    #    print(mdl.get_constraint_by_index(i))

    if obj_opt == "mismatch":

        error1 = 0.0
        error2 = 0.0
        total_error = []
        # minimize trait mismatch
        trait_mismatch = 0.0
        for i in range(num_tasks):
            for j in range(n_strategies):
                error2 = 0.0
                for k in range(n_traits):
                    error2 += ((Y_desired[i][j, k] ** 2) - (
                            2 * mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]) *
                            Y_desired[i][j, k]))  # trait mismatch matrix wrt Strategy i
                trait_mismatch_all.append(error2)

        for i in range(num_tasks):
            error1 = 0.0
            for j in range(n_traits):
                error1 += (mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[j, :]) ** 2)
            total_error.append(error1)

        # Have to split the error expression
        for i in range(num_tasks):
            for j in range(n_strategies):
                trait_mismatch += Z_target[i, j] * trait_mismatch_all[
                    i * n_strategies + j]  # only count if the strategy is used
            trait_mismatch += total_error[i]

        opt_prob = mdl.minimize(trait_mismatch)

    if obj_opt == "agents":

        total_team = 0
        for i in range(num_tasks * n_target_species):
            total_team += X_sol[i]
        opt_prob = mdl.minimize(total_team)

    # print(mdl.get_objective_expr())
    # print(mdl.print_information())

    # solve for X_target
    mdl.solve(url=None, key=None)  # , "Solve failed"
    # print("status:", mdl.solve_status)

    k = 0
    for i in range(num_tasks):
        for j in range(n_target_species):
            X_target[i, j] = X_sol[k].solution_value
            k += 1

    # mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def all_demos(Y_desired, Q_target, n_target, obj_opt):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # Z_target: a matrix indicating which strategies were chosen for each task (n_tasks x n_strategies)
    # X_target: assignment of each agent in the target team
    # error_target: minimum trait mismatch error achieved
    # X_sol: solved cplex variable

    mdl = Model(name='AllDemos')
    params = mdl.parameters
    params.timelimit = 180
    mdl.apply_parameters()

    trait_mismatch_all = []
    n_strategies = Y_desired[0].shape[0]
    n_target_species, n_traits = Q_target.shape
    # n_target = np.ones(n_target_species) * n_agents_per_species
    num_tasks = len(Y_desired)

    X_target = np.zeros((num_tasks, n_target_species))
    Z_target = np.zeros((num_tasks, n_strategies))

    X_sol = mdl.integer_var_list((num_tasks * n_target_species), name='x')
    Z_sol = mdl.binary_var_list((num_tasks * n_strategies), name='z')

    [mdl.add_constraint(mdl.dot(Z_sol[ni:ni + n_strategies], np.ones(n_strategies)) == 1) for ni in
     range(0, (num_tasks * n_strategies), n_strategies)]

    [mdl.add_constraint(mdl.sum(X_sol[(nj * n_target_species) + i] for nj in range(0, num_tasks)) <= n_target[i]) for i
     in range(0, n_target_species)]

    # To ensure that the traits are at least equal to the desired
    for i in range(num_tasks):
        for j in range(n_strategies):
            for k in range(n_traits):
                mdl.add_constraint(Z_sol[i * n_strategies + j] * Y_desired[i][j, k] <= mdl.dot(
                    X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]))

    if obj_opt == "mismatch":

        error1 = 0.0
        error2 = 0.0
        total_error = []
        # minimize trait mismatch
        trait_mismatch = 0.0
        for i in range(num_tasks):
            for j in range(n_strategies):
                error2 = 0.0
                for k in range(n_traits):
                    error2 += ((Y_desired[i][j, k] ** 2) - (
                            2 * mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]) *
                            Y_desired[i][j, k]))  # trait mismatch matrix wrt Strategy i
                trait_mismatch_all.append(error2)

        for i in range(num_tasks):
            error1 = 0.0
            for j in range(n_traits):
                error1 += (mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[j, :]) ** 2)
            total_error.append(error1)

        # Have to split the error expression
        for i in range(num_tasks):
            for j in range(n_strategies):
                trait_mismatch += Z_sol[i * n_strategies + j] * trait_mismatch_all[
                    i * n_strategies + j]  # only count if the strategy is used
            trait_mismatch += total_error[i]

        opt_prob = mdl.minimize(trait_mismatch)

    if obj_opt == "agents":

        total_team = 0
        for i in range(num_tasks * n_target_species):
            total_team += X_sol[i]
        opt_prob = mdl.minimize(total_team)

    # print(mdl.get_objective_expr())
    # print(mdl.print_information())

    # solve for X_target
    mdl.set_time_limit(180)
    mdl.solve(Timelimit=180, url=None, key=None)
    # print("status:", mdl.solve_status)

    k = 0
    for i in range(num_tasks):
        for j in range(n_target_species):
            X_target[i, j] = X_sol[k].solution_value
            k += 1
    t = 0
    for i in range(num_tasks):
        for j in range(n_strategies):
            Z_target[i, j] = Z_sol[t].solution_value
            t += 1
    # mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def all_demos_wc(Y_desired, Q_target, n_target, obj_opt):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # Z_target: a matrix indicating which strategies were chosen for each task (n_tasks x n_strategies)
    # X_target: assignment of each agent in the target team
    # error_target: minimum trait mismatch error achieved
    # X_sol: solved cplex variable

    mdl = Model(name='AllDemosWC')
    params = mdl.parameters
    params.timelimit = 180
    mdl.apply_parameters()

    trait_mismatch_all = []
    n_strategies = Y_desired[0].shape[0]
    n_target_species, n_traits = Q_target.shape
    # n_target = np.ones(n_target_species) * n_agents_per_species
    num_tasks = len(Y_desired)

    X_target = np.zeros((num_tasks, n_target_species))
    Z_target = np.zeros((num_tasks, n_strategies))

    X_sol = mdl.integer_var_list((num_tasks * n_target_species), name='x')
    Z_sol = mdl.binary_var_list((num_tasks * n_strategies), name='z')

    [mdl.add_constraint(mdl.dot(Z_sol[ni:ni + n_strategies], np.ones(n_strategies)) == 1) for ni in
     range(0, (num_tasks * n_strategies), n_strategies)]

    [mdl.add_constraint(mdl.sum(X_sol[(nj * n_target_species) + i] for nj in range(0, num_tasks)) <= n_target[i]) for i
     in range(0, n_target_species)]

    # To ensure that the traits are at least equal to the desired
    """for i in range(num_tasks):
        for j in range(n_strategies):
            for k in range(n_traits):
                mdl.add_constraint(Z_sol[i * n_strategies + j] * Y_desired[i][j, k] <= mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]))"""

    if obj_opt == "mismatch":

        error1 = 0.0
        error2 = 0.0
        total_error = []
        # minimize trait mismatch
        trait_mismatch = 0.0
        for i in range(num_tasks):
            for j in range(n_strategies):
                error2 = 0.0
                for k in range(n_traits):
                    error2 += ((Y_desired[i][j, k] ** 2) - (
                            2 * mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]) *
                            Y_desired[i][j, k]))  # trait mismatch matrix wrt Strategy i
                trait_mismatch_all.append(error2)

        for i in range(num_tasks):
            error1 = 0.0
            for j in range(n_traits):
                error1 += (mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[j, :]) ** 2)
            total_error.append(error1)

        # Have to split the error expression
        for i in range(num_tasks):
            for j in range(n_strategies):
                trait_mismatch += Z_sol[i * n_strategies + j] * trait_mismatch_all[
                    i * n_strategies + j]  # only count if the strategy is used
            trait_mismatch += total_error[i]

        opt_prob = mdl.minimize(trait_mismatch)

    if obj_opt == "agents":

        total_team = 0
        for i in range(num_tasks * n_target_species):
            total_team += X_sol[i]
        opt_prob = mdl.minimize(total_team)

    # print(mdl.get_objective_expr())
    # print(mdl.print_information())

    # solve for X_target
    mdl.set_time_limit(180)
    assert mdl.solve(Timelimit=180, url=None, key=None), "Solve failed"
    # print("status:", mdl.solve_status)
    k = 0
    for i in range(num_tasks):
        for j in range(n_target_species):
            X_target[i, j] = X_sol[k].solution_value
            k += 1
    t = 0
    for i in range(num_tasks):
        for j in range(n_strategies):
            Z_target[i, j] = Z_sol[t].solution_value
            t += 1
    # mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def multi_strategy_transfer(Y_desired, Q_target, n_target, obj_opt):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # Z_target: a matrix indicating which strategies were chosen for each task (n_tasks x n_strategies)
    # X_target: assignment of each agent in the target team
    # error_target: minimum trait mismatch error achieved
    # X_sol: solved cplex variable

    mdl = Model(name='TaskAssignment')

    trait_mismatch_all = []
    n_strategies = Y_desired[0].shape[0]
    n_target_species, n_traits = Q_target.shape
    # n_target = np.ones(n_target_species) * n_agents_per_species
    num_tasks = len(Y_desired)

    X_target = np.zeros((num_tasks, n_target_species))
    Z_target = np.zeros((num_tasks, n_strategies))

    X_sol = mdl.integer_var_list((num_tasks * n_target_species), name='x')
    Z_sol = mdl.binary_var_list((num_tasks * n_strategies), name='z')

    [mdl.add_constraint(mdl.dot(Z_sol[ni:ni + n_strategies], np.ones(n_strategies)) == 1) for ni in
     range(0, (num_tasks * n_strategies), n_strategies)]

    [mdl.add_constraint(mdl.sum(X_sol[(nj * n_target_species) + i] for nj in range(0, num_tasks)) <= n_target[i]) for i
     in range(0, n_target_species)]

    # To ensure that the traits are at least equal to the desired
    for i in range(num_tasks):
        for j in range(n_strategies):
            for k in range(n_traits):
                mdl.add_constraint(Z_sol[i * n_strategies + j] * Y_desired[i][j, k] <= mdl.dot(
                    X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]))

    if obj_opt == "mismatch":

        error1 = 0.0
        error2 = 0.0
        total_error = []
        # minimize trait mismatch
        trait_mismatch = 0.0
        for i in range(num_tasks):
            for j in range(n_strategies):
                error2 = 0.0
                for k in range(n_traits):
                    error2 += ((Y_desired[i][j, k] ** 2) - (
                            2 * mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]) *
                            Y_desired[i][j, k]))  # trait mismatch matrix wrt Strategy i
                trait_mismatch_all.append(error2)

        for i in range(num_tasks):
            error1 = 0.0
            for j in range(n_traits):
                error1 += (mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[j, :]) ** 2)
            total_error.append(error1)

        # Have to split the error expression
        for i in range(num_tasks):
            for j in range(n_strategies):
                trait_mismatch += Z_sol[i * n_strategies + j] * trait_mismatch_all[
                    i * n_strategies + j]  # only count if the strategy is used
            trait_mismatch += total_error[i]

        opt_prob = mdl.minimize(trait_mismatch)

    if obj_opt == "agents":

        total_team = 0
        for i in range(num_tasks * n_target_species):
            total_team += X_sol[i]
        opt_prob = mdl.minimize(total_team)

    # print(mdl.get_objective_expr())
    # print(mdl.print_information())

    # solve for X_target
    mdl.solve(url=None, key=None)
    # print("status:", mdl.solve_status)

    k = 0
    for i in range(num_tasks):
        for j in range(n_target_species):
            X_target[i, j] = X_sol[k].solution_value
            k += 1
    t = 0
    for i in range(num_tasks):
        for j in range(n_strategies):
            Z_target[i, j] = Z_sol[t].solution_value
            t += 1
    # mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def multi_strategy_transfer_wc(Y_desired, Q_target, n_target, obj_opt):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # Z_target: a matrix indicating which strategies were chosen for each task (n_tasks x n_strategies)
    # X_target: assignment of each agent in the target team
    # error_target: minimum trait mismatch error achieved
    # X_sol: solved cplex variable

    mdl = Model(name='TaskAssignmentWC')

    trait_mismatch_all = []
    n_strategies = Y_desired[0].shape[0]
    n_target_species, n_traits = Q_target.shape
    # n_target = np.ones(n_target_species) * n_agents_per_species
    num_tasks = len(Y_desired)

    X_target = np.zeros((num_tasks, n_target_species))
    Z_target = np.zeros((num_tasks, n_strategies))

    X_sol = mdl.integer_var_list((num_tasks * n_target_species), name='x')
    Z_sol = mdl.binary_var_list((num_tasks * n_strategies), name='z')

    [mdl.add_constraint(mdl.dot(Z_sol[ni:ni + n_strategies], np.ones(n_strategies)) == 1) for ni in
     range(0, (num_tasks * n_strategies), n_strategies)]

    [mdl.add_constraint(mdl.sum(X_sol[(nj * n_target_species) + i] for nj in range(0, num_tasks)) <= n_target[i]) for i
     in range(0, n_target_species)]

    # To ensure that the traits are at least equal to the desired
    """for i in range(num_tasks):
        for j in range(n_strategies):
            for k in range(n_traits):
                mdl.add_constraint(Z_sol[i * n_strategies + j] * Y_desired[i][j, k] <= mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]))"""

    if obj_opt == "mismatch":

        error1 = 0.0
        error2 = 0.0
        total_error = []
        # minimize trait mismatch
        trait_mismatch = 0.0
        for i in range(num_tasks):
            for j in range(n_strategies):
                error2 = 0.0
                for k in range(n_traits):
                    error2 += ((Y_desired[i][j, k] ** 2) - (
                            2 * mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[k, :]) *
                            Y_desired[i][j, k]))  # trait mismatch matrix wrt Strategy i
                trait_mismatch_all.append(error2)

        for i in range(num_tasks):
            error1 = 0.0
            for j in range(n_traits):
                error1 += (mdl.dot(X_sol[i * n_target_species:(i + 1) * n_target_species], Q_target.T[j, :]) ** 2)
            total_error.append(error1)

        # Have to split the error expression
        for i in range(num_tasks):
            for j in range(n_strategies):
                trait_mismatch += Z_sol[i * n_strategies + j] * trait_mismatch_all[
                    i * n_strategies + j]  # only count if the strategy is used
            trait_mismatch += total_error[i]

        opt_prob = mdl.minimize(trait_mismatch)

    if obj_opt == "agents":

        total_team = 0
        for i in range(num_tasks * n_target_species):
            total_team += X_sol[i]
        opt_prob = mdl.minimize(total_team)

    # print(mdl.get_objective_expr())
    # print(mdl.print_information())

    # solve for X_target
    assert mdl.solve(url=None, key=None), "Solve failed"
    # print("status:", mdl.solve_status)
    k = 0
    for i in range(num_tasks):
        for j in range(n_target_species):
            X_target[i, j] = X_sol[k].solution_value
            k += 1
    t = 0
    for i in range(num_tasks):
        for j in range(n_strategies):
            Z_target[i, j] = Z_sol[t].solution_value
            t += 1
    # mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def error_calc(Y_desired, Q_test, X_comp, Z_comp):
    num_tasks = len(Y_desired)
    num_strategies, n_traits = Y_desired[0].shape
    #error = np.zeros(n_tasks)
    #comp = np.matmul(X_comp, Q_test)

    Z_target = np.zeros([num_tasks, num_strategies])
    comp = np.dot(X_comp, Q_test)
    error_min = np.zeros(num_tasks)
    error_exact = np.zeros(num_tasks)

    for i in range(num_tasks):
        find_min = [np.linalg.norm(comp[i, :] - Y_desired[i][0, :]), np.linalg.norm(comp[i, :] - Y_desired[i][1, :]),
                    np.linalg.norm(comp[i, :] - Y_desired[i][2, :])]
        ind = find_min.index(min(find_min))
        Z_target[i, ind] = 1
        error_exact[i] = np.linalg.norm(comp[i, :] - Y_desired[i][ind, :]) / np.linalg.norm(Y_desired[i][ind, :])
        for j in range(num_strategies):
            if comp[i, j] > Y_desired[i][ind, j]:
                error_min[i] += 0
            else:
                error_min[i] += (comp[i, j] - Y_desired[i][ind, j]) ** 2
        error_min[i] = math.sqrt(error_min[i]) / np.linalg.norm(Y_desired[i][ind, :])

    return error_min, error_exact
    """if match_type == "exact":
        for i in range(n_tasks):
            for j in range(n_strategies):
                if Z_comp[i, j]:
                    pos = j
            error[i] = np.linalg.norm(comp[i, :] - Y_strategies[i][pos, :]) / np.linalg.norm(Y_strategies[i][pos, :])
    elif match_type == "minimum":
        for i in range(n_tasks):
            for j in range(n_strategies):
                if Z_comp[i, j]:
                    pos = j
            for j in range(n_strategies):
                if comp[i, j] > Y_strategies[i][pos, j]:
                    error[i] += 0
                else:
                    error[i] += (comp[i, j] - Y_strategies[i][pos, j]) ** 2
            error[i] = math.sqrt(error[i]) / np.linalg.norm(Y_strategies[i][pos, :])
    return error"""


def acc_metric(Z_test, ci_test, test_index):
    n_tasks, n_strategies = Z_test.shape
    val = np.zeros(n_tasks)
    for i in range(n_tasks):
        for j in range(n_strategies):
            if Z_test[i, j]:
                if ci_test[i][test_index] == j:
                    val[i] += 1

    return val


def agent_util(X_target, n_target):
    num_tasks, n_species = X_target.shape
    # n_target = n_agents_per_species * np.ones(n_species)
    return np.sum(X_target) / np.sum(n_target)


def normalize(Q, Y):
    n_species, n_traits = Q.shape
    n_tasks = len(Y)
    n_strategies = Y[0].shape[0]
    for i in range(n_traits):
        high = max(Q[:, i])
        if high != 0:
            Q[:, i] = Q[:, i] / high
        for j in range(n_tasks):
            if high != 0:
                Y[j][:, i] = Y[j][:, i] / high

    return Q, Y


if __name__ == "__main__":

    labelsize = 15
    rcParams['xtick.labelsize'] = labelsize

    # Speed has to be made binary - Threshold = 4
    # Add roach as a species [1, 145, 16, 16, 0, 11.2, 0, 1.43, 3.15, 4, 9]
    t = np.array([[1, 1, 0, 1, 1], [100, 80, 45, 125, 145], [50, 80, 0, 0, 16], [8, 13, 6, 5, 16], [0, 13, 6, 0, 0],
                  [18.6, 9.7, 9.8, 9.3, 11.2], [0, 9.7, 9.8, 0, 0],
                  [(1.0 / 0.86), (1.0 / 1.34), (1.0 / 0.61), (1.0 / 1.07), (1.0 / 1.43)],
                  [0, 1, 0, 0, 0], [0, 6, 5, 6, 4], [9, 10, 9, 10, 9]])

    n_traits = 11
    n_demos = 9

    X1 = np.array([[11, 15], [14, 16], [16, 14], [43, 0], [46, 0], [45, 0], [20, 10], [21, 9], [22, 8]])

    Y1 = np.zeros([n_demos, n_traits])
    for i in range(0, n_demos):
        if i < n_demos/3:
            Y1[i, :] = X1[i, 0] * t[0:n_traits, 0] + X1[i, 1] * t[0:n_traits, 1]
        elif i < 2 * n_demos/3:
            Y1[i, :] = X1[i, 0] * t[0:n_traits, 2] + X1[i, 1] * t[0:n_traits, 3]
        else:
            Y1[i, :] = X1[i, 0] * t[0:n_traits, 0] + X1[i, 1] * t[0:n_traits, 3]
    # Hierarchical Clustering
    n_strategies = 3
    cluster = AgglomerativeClustering(n_clusters=n_strategies, affinity='euclidean', linkage='single')
    c = cluster.fit_predict(Y1)
    new_Y1 = np.zeros([n_strategies, n_traits])
    count = np.zeros(n_strategies)
    for j in range(0, n_demos):
        if c[j] == 0:
            new_Y1[0, :] += Y1[j, :]
            count[0] += 1
        elif c[j] == 1:
            new_Y1[1, :] += Y1[j, :]
            count[1] += 1
        elif c[j] == 2:
            new_Y1[2, :] += Y1[j, :]
            count[2] += 1
    mean1 = np.array([new_Y1[0] / count[0], new_Y1[1] / count[1], new_Y1[2] / count[2]])

    Y_strategies = []
    Y_all = []
    Y_all.append(Y1)
    Y_all.append(Y1)
    Y_all.append(Y1)
    Y_all.append(Y1)
    Y_strategies.append(mean1)
    Y_strategies.append(mean1)
    Y_strategies.append(mean1)
    Y_strategies.append(mean1)
    print(Y_strategies)

    Q = []
    # 2 Species possibilities
    temp = np.zeros([n_traits, 2])
    Q.append(t[0:n_traits, 0:2])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 2]
    Q.append(temp)
    temp = np.zeros([n_traits, 2])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 3]
    Q.append(temp)
    temp = np.zeros([n_traits, 2])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 4]
    Q.append(temp)
    Q.append(t[0:n_traits, 1:3])
    temp = np.zeros([n_traits, 2])
    temp[:, 0] = t[0:n_traits, 1]
    temp[:, 1] = t[0:n_traits, 3]
    Q.append(temp)
    temp = np.zeros([n_traits, 2])
    temp[:, 0] = t[0:n_traits, 1]
    temp[:, 1] = t[0:n_traits, 4]
    Q.append(temp)
    Q.append(t[0:n_traits, 2:4])
    temp = np.zeros([n_traits, 2])
    temp[:, 0] = t[0:n_traits, 2]
    temp[:, 1] = t[0:n_traits, 4]
    Q.append(temp)
    temp = np.zeros([n_traits, 2])
    temp[:, 0] = t[0:n_traits, 3]
    temp[:, 1] = t[0:n_traits, 4]
    Q.append(temp)

    # 3 Species Possibilities
    temp = np.zeros([n_traits, 3])
    Q.append(t[0:n_traits, 0:3])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 1]
    temp[:, 2] = t[0:n_traits, 3]
    Q.append(temp)
    temp = np.zeros([n_traits, 3])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 1]
    temp[:, 2] = t[0:n_traits, 4]
    Q.append(temp)
    temp = np.zeros([n_traits, 3])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 2]
    temp[:, 2] = t[0:n_traits, 3]
    Q.append(temp)
    temp = np.zeros([n_traits, 3])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 2]
    temp[:, 2] = t[0:n_traits, 4]
    Q.append(temp)
    temp = np.zeros([n_traits, 3])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 3]
    temp[:, 2] = t[0:n_traits, 4]
    Q.append(temp)
    Q.append(t[0:n_traits, 1:4])
    temp[:, 0] = t[0:n_traits, 1]
    temp[:, 1] = t[0:n_traits, 2]
    temp[:, 2] = t[0:n_traits, 4]
    Q.append(temp)
    temp = np.zeros([n_traits, 3])
    temp[:, 0] = t[0:n_traits, 1]
    temp[:, 1] = t[0:n_traits, 3]
    temp[:, 2] = t[0:n_traits, 4]
    Q.append(temp)
    Q.append(t[0:n_traits, 2:5])

    temp = np.zeros([n_traits, 4])
    Q.append(t[0:n_traits, 0:4])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 1]
    temp[:, 2] = t[0:n_traits, 2]
    temp[:, 3] = t[0:n_traits, 4]
    Q.append(temp)
    temp = np.zeros([n_traits, 4])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 1]
    temp[:, 2] = t[0:n_traits, 3]
    temp[:, 3] = t[0:n_traits, 4]
    Q.append(temp)
    temp = np.zeros([n_traits, 4])
    temp[:, 0] = t[0:n_traits, 0]
    temp[:, 1] = t[0:n_traits, 2]
    temp[:, 2] = t[0:n_traits, 3]
    temp[:, 3] = t[0:n_traits, 4]
    Q.append(temp)
    Q.append(t[0:n_traits, 1:5])
    Q.append(t[0:n_traits, 0:5])

    X_alg = []
    X_b1 = []
    X_b2 = []
    X_b3 = []
    X_b4 = []

    iter_test = np.array([0, 1, 2, 3, 4, 10, 11, 12, 20, 21])
    n_test = 10
    n_dataset = 1
    n_tasks = 4

    error_alg_min = np.zeros([n_dataset, n_test, n_tasks])
    error_alg_exact = np.zeros([n_dataset, n_test, n_tasks])

    util_alg = np.zeros([n_dataset, n_test])
    time_alg = np.zeros([n_dataset, n_test])

    error_b1_min = np.zeros([n_dataset, n_test, n_tasks])
    error_b1_exact = np.zeros([n_dataset, n_test, n_tasks])

    util_b1 = np.zeros([n_dataset, n_test])
    time_b1 = np.zeros([n_dataset, n_test])

    error_b2_min = np.zeros([n_dataset, n_test, n_tasks])
    error_b2_exact = np.zeros([n_dataset, n_test, n_tasks])

    util_b2 = np.zeros([n_dataset, n_test])
    time_b2 = np.zeros([n_dataset, n_test])

    error_b3_min = np.zeros([n_dataset, n_test, n_tasks])
    error_b3_exact = np.zeros([n_dataset, n_test, n_tasks])

    util_b3 = np.zeros([n_dataset, n_test])
    time_b3 = np.zeros([n_dataset, n_test])

    error_b4_min = np.zeros([n_dataset, n_test, n_tasks])
    error_b4_exact = np.zeros([n_dataset, n_test, n_tasks])

    util_b4 = np.zeros([n_dataset, n_test])
    time_b4 = np.zeros([n_dataset, n_test])

    for point in range(n_dataset):
        for test in range(n_test):
            print("Target team index", test)
            Q_target = Q[iter_test[test]].T
            """print("Team before normalization", Q_target)
            Q_target, Y_strategies = normalize(Q_target, Y_strategies)
            print("Team after normalization", Q_target)
            print(Y_strategies)
            if i < 5:
                for k in range(n_species_test):
                    if k == 0 or k == 1 or k == 2:
                        temp = np.random.randint(3, 8)
                        n_target[k] = temp
            elif i < 10:
                for k in range(n_species_test):
                    if k == 1 or k == 2 or k == 4:
                        temp = np.random.randint(3, 8)
                        n_target[k] = temp
            elif i < 15:
                for k in range(n_species_test):
                    if k == 2 or k == 4:
                        temp = np.random.randint(3, 8)
                        n_target[k] = temp
            else:
                for k in range(n_species_test):
                    if k == 2 or k == 4:
                        temp = np.random.randint(3, 8)
                        n_target[k] = temp"""
            if test < 5:
                n_target = [70, 50]
            elif test < 8:
                n_target = [35, 40, 45]
            else:
                n_target = [35, 25, 25, 35]
            start1 = time.process_time()
            try:
                X_target0, Z_target = multi_strategy_transfer(Y_strategies, Q_target, n_target, "mismatch")
            except:
                X_target0, Z_target = multi_strategy_transfer_wc(Y_strategies, Q_target, n_target, "mismatch")
            time_alg[point, test] = time.process_time() - start1
            print('Algorithm' + str(test) + '\n', X_target0)
            X_alg.append(X_target0)
            error_alg_min[point, test, :], error_alg_exact[point, test, :] = error_calc(Y_strategies, Q_target, X_target0, Z_target)
            #error_alg_exact[point, test, :] = error_calc(Y_strategies, Q_target, X_target0, Z_target, "exact")
            util_alg[point, test] = agent_util(X_target0, n_target)

            start2 = time.process_time()
            error_b1_min[point, test, :], error_b1_exact[point, test, :], X_target1, Z_target = global_baseline(
                Y_strategies, Q_target, n_target)
            time_b1[point, test] = time.process_time() - start2
            print('Baseline 1' + str(test) + '\n', X_target1)
            X_b1.append(X_target1)
            util_b1[point, test] = agent_util(X_target1, n_target)

            start3 = time.process_time()
            try:
                X_target2, Z_target = baseline(Y_strategies, Q_target, n_target, "mismatch")
            except:
                X_target2, Z_target = baseline_wc(Y_strategies, Q_target, n_target, "mismatch")
            time_b2[point, test] = time.process_time() - start3
            print('Baseline 2' + str(test) + '\n', X_target2)
            X_b2.append(X_target2)
            error_b2_min[point, test, :], error_b2_exact[point, test, :] = error_calc(Y_strategies, Q_target, X_target2, Z_target)
            #error_b2_exact[point, test, :] = error_calc(Y_strategies, Q_target, X_target2, Z_target, "exact")
            util_b2[point, test] = agent_util(X_target2, n_target)

            #error_b3_min[point, test, :], error_b3_exact[point, test, :], X_target3, Z_target = rand_assign_baseline(
            #    Y_strategies, Q_target, n_target, test)
            start4 = time.process_time()
            error_b3_min[point, test, :], error_b3_exact[point, test, :], X_target3, Z_target = CreateRobotDistribution(
                Y_strategies, Q_target, n_target)
            time_b3[point, test] = time.process_time() - start4
            print('Baseline 3' + str(test) + '\n', X_target3)
            X_b3.append(X_target3)
            util_b3[point, test] = agent_util(X_target3, n_target)

            start5 = time.process_time()
            try:
                X_target5, Z_target = all_demos(Y_all, Q_target, n_target, "mismatch")
            except:
                X_target5, Z_target = all_demos_wc(Y_all, Q_target, n_target, "mismatch")
            time_b4[point, test] = time.process_time() - start5
            print('Baseline 4' + str(test) + '\n', X_target5)
            X_b4.append(X_target0)
            error_b4_min[point, test, :], error_b4_exact[point, test, :] = error_calc(Y_strategies, Q_target, X_target5, Z_target)
            #error_b4_exact[point, test, :] = error_calc(Y_strategies, Q_target, X_target5, Z_target, "exact")
            util_b4[point, test] = agent_util(X_target5, n_target)

    axs = plt.figure(figsize=(14, 10)).subplots(1, 3)
    data1 = [100 * error_alg_min.ravel(order='C'), 100 * error_b1_min.ravel(order='C'),
             100 * error_b2_min.ravel(order='C'), 100 * error_b3_min.ravel(order='C'), 100 * error_b4_min.ravel(order='C')]
    box1 = axs[0].boxplot(data1, widths=0.35, patch_artist=True, boxprops=dict(facecolor='C4'), medianprops=dict(color='black'), showmeans=True,
                          meanprops={"marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"})
    colors = ['blue', 'orange', 'green', 'red', 'yellow']
    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
    axs[0].set_title('Minimum Matching Error Across all Tasks')
    # axs[i, 0].set_xlabel('Our Approach', 'One Strategy', 'Random')
    axs[0].set_xticks([1, 2, 3, 4, 5])
    axs[0].set_xticklabels(['Our\nApproach', 'Unimodal\nStrategy', 'Random\nStrategy', 'Random\nAssignment', 'No\nAbstraction'])
    axs[0].set_ylabel('Minimum Trait Mismatch Error')
    data2 = [100 * error_alg_exact.ravel(order='C'), 100 * error_b1_exact.ravel(order='C'),
             100 * error_b2_exact.ravel(order='C'), 100 * error_b3_exact.ravel(order='C'), 100 * error_b4_exact.ravel(order='C')]
    box2 = axs[1].boxplot(data2, widths=0.35, patch_artist=True, boxprops=dict(facecolor='C4'),
                          medianprops=dict(color='black'), showmeans=True,
                          meanprops={"marker": "*", "markerfacecolor": "black", "markeredgecolor": "black"})
    colors = ['blue', 'orange', 'green', 'red', 'yellow']
    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)
    axs[1].set_title('Exact Matching Error Across all Tasks')
    # axs[i, 0].set_xlabel('Our Approach', 'One Strategy', 'Random')
    axs[1].set_xticks([1, 2, 3, 4, 5])
    axs[1].set_xticklabels(['Our\nApproach', 'Unimodal\nStrategy', 'Random\nStrategy', 'Random\nAssignment',
                            'No\nAbstraction'])
    axs[1].set_ylabel('Exact Trait Mismatch Error')
    # axs[i, 0].legend(data, ('Our Approach', 'Single Strategy Baseline', 'Random Baseline'), loc='upper left')
    data3 = [100 * util_alg.ravel(order='C'), 100 * util_b1.ravel(order='C'), 100 * util_b2.ravel(order='C'),
             100 * util_b3.ravel(order='C'), 100 * util_b4.ravel(order='C')]
    box3 = axs[2].boxplot(data3, widths=0.35, patch_artist=True, boxprops=dict(facecolor='C4'), medianprops=dict(color='black'),
                          showmeans=True, meanprops={"marker": "*", "markerfacecolor": "black", "markeredgecolor": "black"})
    colors = ['blue', 'orange', 'green', 'red', 'yellow']
    for patch, color in zip(box3['boxes'], colors):
        patch.set_facecolor(color)
    axs[2].set_title('Total Agent Utilization')
    axs[2].set_xticks([1, 2, 3, 4, 5])
    axs[2].set_xticklabels(['Our\nApproach', 'Unimodal\nStrategy', 'Random\nStrategy', 'Random\nAssignment', 'No\nAbstraction'])
    axs[2].set_ylabel('Agent Utilization')
    # axs[i, 1].legend(loc='upper left')
    plt.show()

    avg_error_alg_exact = np.zeros(n_tasks)
    avg_error_alg_min = np.zeros(n_tasks)
    avg_error_b1_exact = np.zeros(n_tasks)
    avg_error_b1_min = np.zeros(n_tasks)
    avg_error_b2_exact = np.zeros(n_tasks)
    avg_error_b2_min = np.zeros(n_tasks)
    avg_error_b3_exact = np.zeros(n_tasks)
    avg_error_b3_min = np.zeros(n_tasks)

    std_error_alg_exact = np.zeros(n_tasks)
    std_error_alg_min = np.zeros(n_tasks)
    std_error_b1_exact = np.zeros(n_tasks)
    std_error_b1_min = np.zeros(n_tasks)
    std_error_b2_exact = np.zeros(n_tasks)
    std_error_b2_min = np.zeros(n_tasks)
    std_error_b3_exact = np.zeros(n_tasks)
    std_error_b3_min = np.zeros(n_tasks)

    avg_util_alg = round(100 * sum(util_alg.ravel(order='C')) / (n_test * n_dataset), 1)
    avg_util_b1 = round(100 * sum(util_b1.ravel(order='C')) / (n_test * n_dataset), 1)
    avg_util_b2 = round(100 * sum(util_b2.ravel(order='C')) / (n_test * n_dataset), 1)
    avg_util_b3 = round(100 * sum(util_b3.ravel(order='C')) / (n_test * n_dataset), 1)
    std_util_alg = np.std(100 * util_alg)
    std_util_b1 = np.std(100 * util_b1)
    std_util_b2 = np.std(100 * util_b2)
    std_util_b3 = round(np.std(100 * util_b3))

    print(avg_util_alg)
    print(avg_util_b1)
    print(avg_util_b2)
    print(avg_util_b3)
    print(std_util_alg)
    print(std_util_b1)
    print(std_util_b2)
    print(std_util_b3)

    # Store the Average Errors
    for task in range(n_tasks):
        avg_error_alg_exact[task] = np.mean(error_alg_exact[:, :, task])
        avg_error_alg_min[task] = np.mean(error_alg_min[:, :, task])
        avg_error_b1_exact[task] = np.mean(error_b1_exact[:, :, task])
        avg_error_b1_min[task] = np.mean(error_b1_min[:, :, task])
        avg_error_b2_exact[task] = np.mean(error_b2_exact[:, :, task])
        avg_error_b2_min[task] = np.mean(error_b2_min[:, :, task])
        avg_error_b3_exact[task] = np.mean(error_b3_exact[:, :, task])
        avg_error_b3_min[task] = np.mean(error_b3_min[:, :, task])
        std_error_alg_exact[task] = np.std(error_alg_exact[:, :, task])
        std_error_alg_min[task] = np.std(error_alg_min[:, :, task])
        std_error_b1_exact[task] = np.std(error_b1_exact[:, :, task])
        std_error_b1_min[task] = np.std(error_b1_min[:, :, task])
        std_error_b2_exact[task] = np.std(error_b2_exact[:, :, task])
        std_error_b2_min[task] = np.std(error_b2_min[:, :, task])
        std_error_b3_exact[task] = np.std(error_b3_exact[:, :, task])
        std_error_b3_min[task] = np.std(error_b3_min[:, :, task])

    store_alg_error = {
        'Task Index': range(1, n_tasks + 1),
        'Average Algorithm Exact Matching Error': avg_error_alg_exact,
        'Average Algorithm Minimum Matching Error': avg_error_alg_min,
        'Average Baseline 1 Exact Matching Error': avg_error_b1_exact,
        'Average Baseline 1 Minimum Matching Error': avg_error_b1_min,
        'Average Baseline 2 Exact Matching Error': avg_error_b2_exact,
        'Average Baseline 2 Minimum Matching Error': avg_error_b2_min,
        'Average Baseline 3 Exact Matching Error': avg_error_b3_exact,
        'Average Baseline 3 Minimum Matching Error': avg_error_b3_min,
        'Std Algorithm Exact Matching Error': std_error_alg_exact,
        'Std Algorithm Minimum Matching Error': std_error_alg_min,
        'Std Baseline 1 Exact Matching Error': std_error_b1_exact,
        'Std Baseline 1 Minimum Matching Error': std_error_b1_min,
        'Std Baseline 2 Exact Matching Error': std_error_b2_exact,
        'Std Baseline 2 Minimum Matching Error': std_error_b2_min,
        'Std Baseline 3 Exact Matching Error': std_error_b3_exact,
        'Std Baseline 3 Minimum Matching Error': std_error_b3_min
    }

    all_err = {
        'Alg Min': error_alg_min.ravel(order='C'),
        'Alg Exact': error_alg_exact.ravel(order='C'),
        'B1 Min': error_b1_min.ravel(order='C'),
        'B1 Exact': error_b1_exact.ravel(order='C'),
        'B2 Min': error_b2_min.ravel(order='C'),
        'B2 Exact': error_b2_exact.ravel(order='C'),
        'B3 Min': error_b3_min.ravel(order='C'),
        'B3 Exact': error_b3_exact.ravel(order='C'),
        'B4 Min': error_b4_min.ravel(order='C'),
        'B4 Exact': error_b4_exact.ravel(order='C')
    }

    dferr = pd.DataFrame(all_err, columns=['Alg Min', 'Alg Exact', 'B1 Min', 'B1 Exact', 'B2 Min', 'B2 Exact',
                                           'B3 Min', 'B3 Exact', 'B4 Min', 'B4 Exact'])

    all_util = {
        'Alg util': util_alg.ravel(order='C'),
        'B1 util': util_b1.ravel(order='C'),
        'B2 util': util_b2.ravel(order='C'),
        'B3 util': util_b3.ravel(order='C'),
        'B4 util': util_b4.ravel(order='C')
    }

    dfutil = pd.DataFrame(all_util, columns=['Alg util', 'B1 util', 'B2 util', 'B3 util', 'B4 util'])
    dfutil.to_excel(r'C:\Users\anush\Downloads\cluster_sc2_utilfin.xlsx', index=False, header=True)

    all_time = {
        'Alg time': time_alg.ravel(order='C'),
        'B1 time': time_b2.ravel(order='C'),
        'B2 time': time_b1.ravel(order='C'),
        'B3 time': time_b3.ravel(order='C'),
        'B4 time': time_b4.ravel(order='C')
    }
    dftime = pd.DataFrame(all_time, columns=['Alg time', 'B1 time', 'B2 time', 'B3 time', 'B4 time'])

    dfalg = pd.DataFrame(store_alg_error, columns=['Task Index', 'Average Algorithm Exact Matching Error',
                                                   'Average Algorithm Minimum Matching Error',
                                                   'Average Baseline 1 Exact Matching Error',
                                                   'Average Baseline 1 Minimum Matching Error',
                                                   'Average Baseline 2 Exact Matching Error',
                                                   'Average Baseline 2 Minimum Matching Error',
                                                   'Average Baseline 3 Exact Matching Error',
                                                   'Average Baseline 3 Minimum Matching Error',
                                                   'Std Algorithm Exact Matching Error',
                                                   'Std Algorithm Minimum Matching Error',
                                                   'Std Baseline 1 Exact Matching Error',
                                                   'Std Baseline 1 Minimum Matching Error',
                                                   'Std Baseline 2 Exact Matching Error',
                                                   'Std Baseline 2 Minimum Matching Error',
                                                   'Std Baseline 3 Exact Matching Error',
                                                   'Std Baseline 3 Minimum Matching Error'])
    dfalg.set_index('Task Index', inplace=True)
    print(dfalg)