"""
   Numerical Simulations
   Experiment to use clustered data from expert demonstrations
   and compute minimum and exact matching errors
"""

import numpy as np
import cvxpy
import math
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import pandas as pd
from docplex.mp.advmodel import Model
import time
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import AgglomerativeClustering


def CreateRobotDistribution(Y_desired, Q_target, n_robot_per_species, start, task_restrict=None):
    num_tasks = len(Y_desired)
    num_strategies = Y_desired[0].shape[0]
    n_species = Q_target.shape[0]
    if task_restrict is None:
        task_restrict = range(num_tasks)
    X_target = np.zeros((num_tasks, n_species))
    for s in range(n_species):
        R = np.random.choice(task_restrict, size=n_robot_per_species)
        for m in task_restrict:
            X_target[m, s] = np.sum(R == m)
    time_taken = time.process_time() - start

    Z_target = np.zeros([num_tasks, num_strategies])
    comp = np.dot(X_target, Q_target)
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
    return error_min, error_exact, X_target.astype(np.int32), Z_target, time_taken


def cluster_creation(n_samples=100, n_clusters=2, task_index=0):
    """
        Create Y clusters using make_blobs - for a single task
    """
    # random_state = 10
    while True:
        varied, y, center = make_blobs(n_samples=n_samples, n_features=3, centers=n_clusters, center_box=(20, 40),
                                   cluster_std=[1.0, 0.5, 0.25], random_state=task_index, return_centers=True)
        flag = 0
        for i in range(0, 3):
            c1 = 0
            c2 = 0
            for j in range(0, 3):
                if (center[i, j] - center[(i+1) % 3, j]) > 0:
                    c1 += 1
                else:
                    c2 += 1
            if c1 != 3 and c2 != 3:
                flag += 1
        if flag == 3:
            break
        task_index = np.random.randint(0, 100)


    # Split the generated data into training and testing
    Y_train, Y_test = train_test_split(varied, test_size=0.2, shuffle=False)
    ci_train, ci_test = train_test_split(y, test_size=0.2, shuffle=False)
   
    return Y_train, Y_test, ci_train, ci_test


def species_creation(clusters, sample, n_tasks, n_species):
    # inputs:
    # clusters: The generated Y samples for each task
    # outputs:
    # Q_target: The species-trait matrix for the sample

    #num_species = len(clusters)
    num_species = n_species
    num_traits = 3
    Q_target = np.zeros([num_species, num_traits])

    for i in range(num_species):
        task_index = np.random.randint(0, n_tasks)
        #print(sample)
        sample = np.random.randint(0, 3)
        temp = np.random.randint(10, 25)
        for k in range(num_traits):
            Q_target[i, k] = abs(clusters[task_index][sample, k] / temp)
    return Q_target


def specieswise_transfer(Y_cluster, sample_index, Q_target, n_agents_per_species, exact_match):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # X_target: assignment of each agent in the target team

    # num_tasks = Y_desired[0].shape[0]
    num_tasks = len(Y_cluster)
    n_traits = Y_cluster[0].shape[1]
    Y_desired = np.zeros([num_tasks, n_traits])

    for i in range(num_tasks):
        Y_desired[i, :] = Y_cluster[i][sample_index, :]
    n_target_species = Q_target.shape[0]
    n_target = np.ones(n_target_species) * n_agents_per_species

    X_sol = cvxpy.Variable((num_tasks, n_target_species), integer=True)

    # minimize trait mismatch
    if exact_match:
        mismatch_mat = Y_desired - cvxpy.matmul(X_sol, Q_target)  # trait mismatch matrix
    else:
        mismatch_mat = cvxpy.pos(Y_desired - cvxpy.matmul(X_sol, Q_target))
    # mismatch_mat = Y_desired - cvxpy.matmul(X_sol, Q_target)  # trait mismatch matrix
    obj = cvxpy.Minimize(cvxpy.pnorm(mismatch_mat, 2))
    # obj = cvxpy.Minimize(cvxpy.sum(cvxpy.multiply(mismatch_mat, mismatch_mat)))

    # ensure each agent is only assigned to one task
    constraints = [cvxpy.matmul(X_sol.T, np.ones([num_tasks, 1])) <= np.array([n_target]).T, X_sol >= 0]

    # solve for X_target
    opt_prob = cvxpy.Problem(obj, constraints)
    opt_prob.solve(solver=cvxpy.CPLEX)
    X_target = X_sol.value
    # print(obj.value)

    return X_target


def rand_assign_baseline(Y_desired, Q_target, n_agents_per_species):
    n_species, n_traits = Q_target.shape
    num_tasks = len(Y_desired)
    num_strategies = Y_desired[0].shape[0]
    n_target = n_agents_per_species * np.ones(n_species)
    X_target = np.zeros((num_tasks, n_species))
    for i in range(num_tasks):
        for j in range(n_species):
            if i == 0:
                X_target[i, j] = np.random.randint(0, n_target[j])
            else:
                X_target[i, j] = n_target[j] - X_target[0, j]
    Z_target = np.zeros([num_tasks, num_strategies])
    comp = np.dot(X_target, Q_target)
    error_min = np.zeros(num_tasks)
    error_exact = np.zeros(num_tasks)
    for i in range(num_tasks):
        find_min = [np.linalg.norm(comp[i, :] - Y_desired[i][0, :]), np.linalg.norm(comp[i, :] - Y_desired[i][1, :]),
                    np.linalg.norm(comp[i, :] - Y_desired[i][2, :])]
        ind = find_min.index(min(find_min))
        Z_target[i, ind] = 1
        error_exact[i] = np.linalg.norm(comp[i, :] - Y_desired[i][ind, :]) / np.linalg.norm(Y_desired[i][ind, :])
        if np.linalg.norm(comp[i, :]) <= np.linalg.norm(Y_desired[i][ind, :]):
            error_min[i] = np.linalg.norm(comp[i, :] - Y_desired[i][ind, :]) / np.linalg.norm(Y_desired[i][ind, :])

    return error_min, error_exact, X_target, Z_target


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
    opt_prob.solve(solver=cvxpy.CPLEX, cplex_params={"timelimit":1800})
    X_target = X_sol.value

    return X_target


def global_baseline(Y_desired, Q_target, n_agents_per_species, start):
    # Compute the centroid for each task
    # Do the task assignment - specieswise transfer

    num_strategies, num_traits = Y_desired[0].shape
    num_tasks = len(Y_desired)
    Y_baseline = np.zeros([num_tasks, num_traits])
    n_species = Q_target.shape[0]
    n_target = n_agents_per_species * np.ones(n_species)

    for i in range(num_tasks):
        temp = np.zeros(num_traits)
        for j in range(num_strategies):
            temp += Y_desired[i][j, :]
        Y_baseline[i, :] = temp / num_strategies
    X_target = baseline_transfer(Y_baseline, Q_target, n_target)
    time_taken = time.process_time() - start

    Z_target = np.zeros([num_tasks, num_strategies])
    comp = np.dot(X_target, Q_target)
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

    return error_min, error_exact, X_target, Z_target, time_taken


def baseline(Y_desired, Q_target, n_agents_per_species, obj_opt):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # Z_target: a matrix indicating which strategies were chosen for each task (n_tasks x n_strategies)
    # X_target: assignment of each agent in the target team
    # error_target: minimum trait mismatch error achieved
    # X_sol: solved cplex variable

    #start = time.process_time()

    mdl = Model('Baseline')
    params = mdl.parameters
    params.timelimit = 1800
    trait_mismatch_all = []
    # num_tasks = Y_desired[0].shape[0]
    num_tasks = len(Y_desired)
    n_target_species, n_traits = Q_target.shape
    n_target = np.ones(n_target_species) * n_agents_per_species
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
    mdl.set_time_limit(1800)
    mdl.solve(url=None, key=None)#, cplex_parameter=dict(optimalitytarget=2))  # , "Solve failed"
    # print("status:", mdl.solve_status)

    k = 0
    for i in range(num_tasks):
        for j in range(n_target_species):
            X_target[i, j] = X_sol[k].solution_value
            k += 1
    #mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()
    #time_taken = time.process_time() - start

    return X_target, Z_target#, time_taken


def baseline_wc(Y_desired, Q_target, n_agents_per_species, obj_opt):
    # inputs:
    # Y_desired: desired capabilities (num_tasks x num_trait)
    # Q_target: target team's trait distribution matrix (n_target x num_trait)
    # n_target: number of agents/species in target team (n_species x 1)

    # outputs:
    # Z_target: a matrix indicating which strategies were chosen for each task (n_tasks x n_strategies)
    # X_target: assignment of each agent in the target team
    # error_target: minimum trait mismatch error achieved
    # X_sol: solved cplex variable

    mdl = Model('BaselineWC')
    params = mdl.parameters
    params.timelimit = 1800
    trait_mismatch_all = []
    # num_tasks = Y_desired[0].shape[0]
    num_tasks = len(Y_desired)
    n_target_species, n_traits = Q_target.shape
    n_target = np.ones(n_target_species) * n_agents_per_species
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
    mdl.set_time_limit(1800)
    mdl.solve(url=None, key=None)#, cplex_parameter=dict(optimalitytarget=2))  # , "Solve failed"
    # print("status:", mdl.solve_status)

    k = 0
    for i in range(num_tasks):
        for j in range(n_target_species):
            X_target[i, j] = X_sol[k].solution_value
            k += 1
    #mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def all_demos_baseline(Y_desired, Q_target, n_agents_per_species, obj_opt):
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
    params.timelimit = 1800

    trait_mismatch_all = []
    n_strategies = Y_desired[0].shape[0]
    n_target_species, n_traits = Q_target.shape
    n_target = np.ones(n_target_species) * n_agents_per_species
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
    mdl.set_time_limit(1800)
    mdl.solve(url=None, key=None)#, cplex_parameters=dict(optimalitytarget=2))
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
    #mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def all_demos_baseline_wc(Y_desired, Q_target, n_agents_per_species, obj_opt):
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
    params.timelimit = 1800
    trait_mismatch_all = []
    n_strategies = Y_desired[0].shape[0]
    n_target_species, n_traits = Q_target.shape
    n_target = np.ones(n_target_species) * n_agents_per_species
    num_tasks = len(Y_desired)

    X_target = np.zeros((num_tasks, n_target_species))
    Z_target = np.zeros((num_tasks, n_strategies))

    X_sol = mdl.integer_var_list((num_tasks * n_target_species), name='x')
    Z_sol = mdl.binary_var_list((num_tasks * n_strategies), name='z')

    [mdl.add_constraint(mdl.dot(Z_sol[ni:ni + n_strategies], np.ones(n_strategies)) == 1) for ni in
     range(0, (num_tasks * n_strategies), n_strategies)]

    [mdl.add_constraint(mdl.sum(X_sol[(nj * n_target_species) + i] for nj in range(0, num_tasks)) <= n_target[i]) for i
     in range(0, n_target_species)]

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
    mdl.set_time_limit(1800)
    mdl.solve(url=None, key=None)#, cplex_parameters=dict(optimalitytarget=2))
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
    #mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def multi_strategy_transfer(Y_desired, Q_target, n_agents_per_species, obj_opt):
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
    params = mdl.parameters
    params.timelimit = 1800
    trait_mismatch_all = []
    n_strategies = Y_desired[0].shape[0]
    n_target_species, n_traits = Q_target.shape
    n_target = np.ones(n_target_species) * n_agents_per_species
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
    mdl.set_time_limit(1800)
    mdl.solve(url=None, key=None)#, cplex_parameters=dict(optimalitytarget=2))
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
    #mdl.print_solution(print_zeros=True)
    # error_target = mdl.objective_value
    mdl.clear()

    return X_target, Z_target


def multi_strategy_transfer_wc(Y_desired, Q_target, n_agents_per_species, obj_opt):
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
    params = mdl.parameters
    params.timelimit = 1800
    trait_mismatch_all = []
    n_strategies = Y_desired[0].shape[0]
    n_target_species, n_traits = Q_target.shape
    n_target = np.ones(n_target_species) * n_agents_per_species
    num_tasks = len(Y_desired)

    X_target = np.zeros((num_tasks, n_target_species))
    Z_target = np.zeros((num_tasks, n_strategies))

    X_sol = mdl.integer_var_list((num_tasks * n_target_species), name='x')
    Z_sol = mdl.binary_var_list((num_tasks * n_strategies), name='z')

    # ensure each agent is only assigned to one task and only one strategy is picked per task
    # constraints = [mdl.dot(X_sol.T, np.ones([num_tasks, 1])) == np.array([n_target]).T, X_sol >= 0,
    #              mdl.sum(Z_sol, axis=1) == np.ones(n_tasks)]

    [mdl.add_constraint(mdl.dot(Z_sol[ni:ni + n_strategies], np.ones(n_strategies)) == 1) for ni in
     range(0, (num_tasks * n_strategies), n_strategies)]

    [mdl.add_constraint(mdl.sum(X_sol[(nj * n_target_species) + i] for nj in range(0, num_tasks)) <= n_target[i]) for i
     in range(0, n_target_species)]

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
    mdl.set_time_limit(1800)
    mdl.solve(url=None, key=None)#, cplex_parameter=dict(optimalitytarget=2))
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
    #mdl.print_solution(print_zeros=True)
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


def agent_util(X_target, n_agents_per_species):
    num_tasks, n_species = X_target.shape
    n_target = n_agents_per_species * np.ones(n_species)
    return np.sum(X_target) / np.sum(n_target)


"""def label_diff(ax, i, j, text, X, Y):
    x = (X[i]+X[j])/2
    y = max(Y[i], Y[j])
    dx = abs(1)
    props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 20, 'shrinkB': 20, 'linewidth': 2}
    ax.annotate(text, xy=(x, y+5), xytext=(x, y+5), zorder=10, ha='center')
    ax.annotate('', xy=(X[i], Y[j]), xytext=(X[j], Y[j]), arrowprops=props)"""


if __name__ == "__main__":

    print('\n--------\n')

    print("\nTesting multi-strategy dataset generation....\n")

    n_traits = 3  # number of traits in the target team
    n_tasks = 3  # number of tasks
    n_agents_per_species = 33
    n_strategies = 3  # number of clusters for each task
    n_samples = 60  # number of samples in each task
    n_train = int(n_samples * 4 / 5)
    n_test = int(n_samples / 5)
    n_dataset = 5

    error_alg_min = np.zeros([n_dataset, n_test, n_tasks])
    error_alg_exact = np.zeros([n_dataset, n_test, n_tasks])
    acc_alg = np.zeros([n_dataset, n_test, n_tasks])
    util_alg = np.zeros([n_dataset, n_test])

    error_b1_min = np.zeros([n_dataset, n_test, n_tasks])
    error_b1_exact = np.zeros([n_dataset, n_test, n_tasks])
    acc_b1 = np.zeros([n_dataset, n_test, n_tasks])
    util_b1 = np.zeros([n_dataset, n_test])

    error_b2_min = np.zeros([n_dataset, n_test, n_tasks])
    error_b2_exact = np.zeros([n_dataset, n_test, n_tasks])
    acc_b2 = np.zeros([n_dataset, n_test, n_tasks])
    util_b2 = np.zeros([n_dataset, n_test])

    error_b3_min = np.zeros([n_dataset, n_test, n_tasks])
    error_b3_exact = np.zeros([n_dataset, n_test, n_tasks])
    acc_b3 = np.zeros([n_dataset, n_test, n_tasks])
    util_b3 = np.zeros([n_dataset, n_test])

    error_b4_min = np.zeros([n_dataset, n_test, n_tasks])
    error_b4_exact = np.zeros([n_dataset, n_test, n_tasks])
    acc_b4 = np.zeros([n_dataset, n_test, n_tasks])
    util_b4 = np.zeros([n_dataset, n_test])

    index = 0
    time_alg = []
    time_b1 = []
    time_b2 = []
    time_b3 = []
    time_b4 = []

    for point in range(n_dataset):
        Y_cluster_train = []
        Y_cluster_test = []
        Y_strategies = []
        Y_demos_baseline = []
        ci_train = []
        ci_test = []

        Q_cluster = []
        X_cluster = []

        # Training Phase
        """if point < 3:
            n_tasks = 2
        elif point < 6:
            n_tasks = 4
        elif point < 9:
            n_tasks = 6
        elif point < 20:
            n_tasks = 8
        elif point < 25:
            n_tasks = 10"""
        if point % 5 == 0:
            index = 0
        index += 1
        n_species = 2 * index
        n_agents_per_species = int(33 / index)
        print(n_agents_per_species)
        for i in range(n_tasks):
            init = np.random.randint(1, 100)
            #init += 1
            t1, t2, c1, c2 = cluster_creation(n_samples, n_strategies, init)
            print("Number of demos", t1.shape)
            val = np.zeros([n_strategies, n_traits])
            for j in range(n_train):
                if c1[j] == 0:
                    val[0, :] += t1[j, :]
                elif c1[j] == 1:
                    val[1, :] += t1[j, :]
                else:
                    val[2, :] += t1[j, :]
	    
	    # Hierarchical Clustering
            cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
            new_c1 = cluster.fit_predict(t1)
            new_val = np.zeros([n_strategies, n_traits])
            for j in range(n_train):
                if new_c1[j] == 0:
                    new_val[0, :] += t1[j, :]
                elif new_c1[j] == 1:
                    new_val[1, :] += t1[j, :]
                else:
                    new_val[2, :] += t1[j, :]
            mean = np.array([val[0] / 16, val[1] / 16, val[2] / 16])
            new_mean = np.array([new_val[0] / 16, new_val[1] / 16, new_val[2] / 16])
            Y_demos_baseline.append(t1)
            Y_strategies.append(new_mean)
            Y_cluster_train.append(t1)
            Y_cluster_test.append(t2)
            ci_train.append(c1)
            ci_test.append(c2)

        for i in range(n_samples):
            if i < n_train:
                #Q_cluster.append(species_creation(Y_cluster_train, i, n_tasks, n_species))
                Q_cluster.append(species_creation(Y_strategies, i, n_tasks, n_species))
                X_cluster.append(specieswise_transfer(Y_cluster_train, i, Q_cluster[i], n_agents_per_species, True))
            else:
                #Q_cluster.append(species_creation(Y_cluster_test, i - n_train, n_tasks, n_species))
                Q_cluster.append(species_creation(Y_strategies, i, n_tasks, n_species))
                X_cluster.append(specieswise_transfer(Y_cluster_test, i - n_train, Q_cluster[i], n_agents_per_species, True))

        # Testing Phase
        # test = 245

        for test in range(48, 60):
            start = time.process_time()
            try:
                #start = time.process_time()
                X_target0, Z_target = multi_strategy_transfer(Y_strategies, Q_cluster[test], n_agents_per_species,
                                                             "mismatch")
                #time_alg.append(time.process_time() - start)
            except:
                #start = time.process_time()
                X_target0, Z_target = multi_strategy_transfer_wc(Y_strategies, Q_cluster[test], n_agents_per_species,
                                                                "mismatch")
            time_alg.append(time.process_time() - start)
            error_alg_min[point, test - 48, :], error_alg_exact[point, test - 48, :] = error_calc(Y_strategies, Q_cluster[test], X_target0, Z_target)
            #error_alg_exact[point, test - 48, :] = error_calc(Y_strategies, Q_cluster[test], X_target0, Z_target, "exact")
            acc_alg[point, test - 48, :] = acc_metric(Z_target, ci_test, test - 48)
            util_alg[point, test - 48] = agent_util(X_target0, n_agents_per_species)
            start2 = time.process_time()
            error_b1_min[point, test - 48, :], error_b1_exact[point, test - 48, :], X_target1, Z_target, finish = global_baseline(Y_strategies, Q_cluster[test],
                                                                                         n_agents_per_species, start2)
            time_b2.append(finish)
            acc_b1[point, test - 48, :] = acc_metric(Z_target, ci_test, test - 48)
            util_b1[point, test - 48] = agent_util(X_target1, n_agents_per_species)
            start1 = time.process_time()
            try:
                #start1 = time.process_time()
                X_target2, Z_target = baseline(Y_strategies, Q_cluster[test], n_agents_per_species, "mismatch")
                #time_b1.append(time.process_time() - start1)
            except:
                #start1 = time.process_time()
                X_target2, Z_target = baseline_wc(Y_strategies, Q_cluster[test], n_agents_per_species, "mismatch")
            time_b1.append(time.process_time() - start1)
            error_b2_min[point, test - 48, :], error_b2_exact[point, test - 48, :] = error_calc(Y_strategies, Q_cluster[test], X_target2, Z_target)
            #error_b2_exact[point, test - 48, :] = error_calc(Y_strategies, Q_cluster[test], X_target2, Z_target, "exact")
            acc_b2[point, test - 48, :] = acc_metric(Z_target, ci_test, test - 48)
            util_b2[point, test - 48] = agent_util(X_target2, n_agents_per_species)

            start3 = time.process_time()
            error_b3_min[point, test - 48, :], error_b3_exact[point, test - 48, :], X_target3, Z_target, time_taken = CreateRobotDistribution(Y_strategies,
                                                                                                 Q_cluster[test],
                                                                                                 n_agents_per_species, start3)
            time_b3.append(time_taken)
            acc_b3[point, test - 48, :] = acc_metric(Z_target, ci_test, test - 48)
            util_b3[point, test - 48] = agent_util(X_target3, n_agents_per_species)
            start4 = time.process_time()
            try:
                # start4 = time.process_time()
                X_target4, Z_target = all_demos_baseline(Y_demos_baseline, Q_cluster[test], n_agents_per_species, "mismatch")
                # time_b4.append(time.process_time() - start4)
            except:
                # start4 = time.process_time()
                X_target4, Z_target = all_demos_baseline_wc(Y_demos_baseline, Q_cluster[test], n_agents_per_species, "mismatch")
            time_b4.append(time.process_time() - start4)
            print("Time taken for B4", time_b4)
            error_b4_min[point, test - 48, :], error_b4_exact[point, test - 48, :] = error_calc(Y_strategies, Q_cluster[test], X_target4, Z_target)
            #error_b2_exact[point, test - 48, :] = error_calc(Y_strategies, Q_cluster[test], X_target2, Z_target, "exact")
            acc_b4[point, test - 48, :] = acc_metric(Z_target, ci_test, test - 48)
            util_b4[point, test - 48] = agent_util(X_target4, n_agents_per_species)

    avg_error_alg_exact = round(100 * sum(error_alg_exact.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_alg_min = round(100 * sum(error_alg_min.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_b1_exact = round(100 * sum(error_b1_exact.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_b1_min = round(100 * sum(error_b1_min.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_b2_exact = round(100 * sum(error_b2_exact.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_b2_min = round(100 * sum(error_b2_min.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_b3_min = round(100 * sum(error_b3_min.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_b3_exact = round(100 * sum(error_b3_exact.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_b4_min = round(100 * sum(error_b4_min.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_error_b4_exact = round(100 * sum(error_b4_exact.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)

    std_error_alg_exact = np.std(error_alg_exact.ravel(order='C'))
    std_error_alg_min = np.std(error_alg_min.ravel(order='C'))
    std_error_b1_exact = np.std(error_b1_exact.ravel(order='C'))
    std_error_b1_min = np.std(error_b1_min.ravel(order='C'))
    std_error_b2_exact = np.std(error_b1_exact.ravel(order='C'))
    std_error_b2_min = np.std(error_b2_min.ravel(order='C'))
    std_error_b3_exact = np.std(error_b3_exact.ravel(order='C'))
    std_error_b3_min = np.std(error_b3_min.ravel(order='C'))
    std_error_b4_exact = np.std(error_b4_exact.ravel(order='C'))
    std_error_b4_min = np.std(error_b4_min.ravel(order='C'))

    avg_acc_alg = round(100 * sum(acc_alg.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_acc_b1 = round(100 * sum(acc_b1.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_acc_b2 = round(100 * sum(acc_b2.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_acc_b3 = round(100 * sum(acc_b3.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)
    avg_acc_b4 = round(100 * sum(acc_b4.ravel(order='C')) / (n_dataset * n_test * n_tasks), 2)

    avg_util_alg = round(100 * sum(util_alg.ravel(order='C')) / (n_dataset * n_test), 2)
    avg_util_b1 = round(100 * sum(util_b1.ravel(order='C')) / (n_dataset * n_test), 2)
    avg_util_b2 = round(100 * sum(util_b2.ravel(order='C')) / (n_dataset * n_test), 2)
    avg_util_b3 = round(100 * sum(util_b3.ravel(order='C')) / (n_dataset * n_test), 2)
    avg_util_b4 = round(100 * sum(util_b4.ravel(order='C')) / (n_dataset * n_test), 2)

    std_acc_alg = np.std(100 * acc_alg.ravel(order='C'))
    std_acc_b1 = np.std(100 * acc_b1.ravel(order='C'))
    std_acc_b2 = np.std(100 * acc_b2.ravel(order='C'))
    std_acc_b3 = np.std(100 * acc_b3.ravel(order='C'))
    std_acc_b4 = np.std(100 * acc_b4.ravel(order='C'))

    std_util_alg = np.std(100 * util_alg.ravel(order='C'))
    std_util_b1 = np.std(100 * util_b1.ravel(order='C'))
    std_util_b2 = np.std(100 * util_b2.ravel(order='C'))
    std_util_b3 = np.std(100 * util_b3.ravel(order='C'))
    std_util_b4 = np.std(100 * util_b4.ravel(order='C'))

    print(error_alg_min.shape)
    print(acc_alg.shape)
    print(util_alg.shape)

    print("Mean Alg", np.mean(time_alg))
    print("Std Alg", np.std(time_alg))
    print("Mean B1", np.mean(time_b1))
    print("Std B1", np.std(time_b1))
    print("Mean B2", np.mean(time_b2))
    print("Std B2", np.std(time_b2))
    print("Mean B3", np.mean(time_b3))
    print("Std B3", np.std(time_b3))
    print("Mean B4", np.mean(time_b4))
    print("Std B4", np.std(time_b4))

    axs = plt.figure(figsize=(12, 10)).subplots(1, 3)
    # for i in range(0, 2):
    data1 = [100 * error_alg_min.ravel(order='C'), 100 * error_b1_min.ravel(order='C'),
             100 * error_b2_min.ravel(order='C'), 100 * error_b3_min.ravel(order='C'), 100 * error_b4_min.ravel(order='C')]
    box1 = axs[0].boxplot(data1, widths=0.35, patch_artist=True, medianprops=dict(color='black'), showmeans=True,
                          meanprops={"marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"})
    colors = ['blue', 'orange', 'green', 'red', 'pink']
    for patch, color in zip(box1['boxes'], colors):
        patch.set_facecolor(color)
    axs[0].set_title('Minimum Matching Error Across all Tasks')
    # axs[i, 0].set_xlabel('Our Approach', 'One Strategy', 'Random')
    axs[0].set_xticks([1, 2, 3, 4, 5])
    axs[0].set_xticklabels(['Our\nApproach', 'Unimodal\nStrategy', 'Random\nStrategy', 'Random\nAssignment', 'All\nDemos'])
    """x = [1, 2, 3, 4]
    #y = [30, 31, 32, 33]
    y = [np.max(data1[0])+20, np.max(data1[1])+20, np.max(data1[2])+20, np.max(data1[3])+20]
    label_diff(axs[0], 0, 1, 'p=1.1e-31', x, y)
    label_diff(axs[0], 0, 2, 'p=2.5e-8', x, y)
    label_diff(axs[0], 0, 3, 'p=4.5e-19', x, y)"""
    axs[0].set_yticks([0, 20, 40, 60, 100])
    axs[0].set_ylabel('Minimum Trait Mismatch Error')
    data2 = [100 * error_alg_exact.ravel(order='C'), 100 * error_b1_exact.ravel(order='C'),
             100 * error_b2_exact.ravel(order='C'), 100 * error_b3_exact.ravel(order='C'), 100 * error_b4_exact.ravel(order='C')]
    box2 = axs[1].boxplot(data2, widths=0.35, patch_artist=True, medianprops=dict(color='black'), showmeans=True,
                          meanprops={"marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"})
    colors = ['blue', 'orange', 'green', 'red', 'pink']
    for patch, color in zip(box2['boxes'], colors):
        patch.set_facecolor(color)
    axs[1].set_title('Exact Matching Error Across all Tasks')
    # axs[i, 0].set_xlabel('Our Approach', 'One Strategy', 'Random')
    axs[1].set_xticks([1, 2, 3, 4, 5])
    axs[1].set_yticks([0, 20, 40, 60, 100])
    axs[1].set_xticklabels(['Our\nApproach', 'Unimodal\nStrategy', 'Random\nStrategy', 'Random\nAssignment', 'All\nDemos'])
    axs[1].set_ylabel('Exact Trait Mismatch Error')
    """x = [1, 2, 3, 4]
    #y = [30, 31, 32, 33]
    y = [np.max(data2[0])+20, np.max(data1[1])+20, np.max(data2[2])+20, np.max(data2[3])+20]
    label_diff(axs[1], 0, 1, 'p=1.1e-31', x, y)
    label_diff(axs[1], 0, 2, 'p=2.5e-8', x, y)
    label_diff(axs[1], 0, 3, 'p=4.5e-19', x, y)"""
    # axs[i, 0].legend(data, ('Our Approach', 'Single Strategy Baseline', 'Random Baseline'), loc='upper left')
    data3 = [100 * util_alg.ravel(order='C'), 100 * util_b1.ravel(order='C'), 100 * util_b2.ravel(order='C'),
             100 * util_b3.ravel(order='C'), 100 * util_b4.ravel(order='C')]
    box3 = axs[2].boxplot(data3, widths=0.35, patch_artist=True, medianprops=dict(color='black'), showmeans=True,
                          meanprops={"marker":"*", "markerfacecolor":"black", "markeredgecolor":"black"})
    colors = ['blue', 'orange', 'green', 'red', 'pink']
    for patch, color in zip(box3['boxes'], colors):
        patch.set_facecolor(color)
    axs[2].set_title('Average Agent Utilization')
    axs[2].set_xticks([1, 2, 3, 4, 5])
    axs[2].set_xticklabels(['Our\nApproach', 'Unimodal\nStrategy', 'Random\nStrategy', 'Random\nAssignment', 'All\nDemos'])
    axs[2].set_ylabel('Agent Utilization')
    # axs[i, 1].legend(loc='upper left')

    plt.show()

    print(avg_util_alg)
    print(avg_util_b1)
    print(avg_util_b2)
    print(avg_util_b3)
    print(avg_util_b4)

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

    all_time = {
        'Alg time': time_alg,
        'B1 time': time_b2,
        'B2 time': time_b1,
        'B3 time': time_b3,
        'B4 time': time_b4
    }
    dftime = pd.DataFrame(all_time, columns=['Alg time', 'B1 time', 'B2 time', 'B3 time', 'B4 time'])

    store_alg_error = {
        'Task Index': range(1),
        'Algorithm Exact Matching Error': avg_error_alg_exact,
        'Algorithm Minimum Matching Error': avg_error_alg_min,
        'Algorithm Accuracy Metric': avg_acc_alg,
        'Baseline 1 Exact Matching Error': avg_error_b1_exact,
        'Baseline 1 Minimum Matching Error': avg_error_b1_min,
        'Baseline 1 Accuracy Metric': avg_acc_b1,
        'Baseline 2 Exact Matching Error': avg_error_b2_exact,
        'Baseline 2 Minimum Matching Error': avg_error_b2_min,
        'Baseline 2 Accuracy Metric': avg_acc_b2,
        'Baseline 3 Exact Matching Error': avg_error_b3_exact,
        'Baseline 3 Minimum Matching Error': avg_error_b3_min,
        'Baseline 3 Accuracy Metric': avg_acc_b3,
        'Baseline 4 Accuracy Metric': avg_acc_b4,
        'Std Algorithm Exact Matching Error': std_error_alg_exact,
        'Std Algorithm Minimum Matching Error': std_error_alg_min,
        'Std Baseline 1 Exact Matching Error': std_error_b1_exact,
        'Std Baseline 1 Minimum Matching Error': std_error_b1_min,
        'Std Baseline 2 Exact Matching Error': std_error_b2_exact,
        'Std Baseline 2 Minimum Matching Error': std_error_b2_min,
        'Std Baseline 3 Exact Matching Error': std_error_b3_exact,
        'Std Baseline 3 Minimum Matching Error': std_error_b3_min,
        'Std Baseline 4 Minimum Matching Error': std_error_b4_min
    }

    dfalg = pd.DataFrame(store_alg_error, columns=['Task Index', 'Algorithm Exact Matching Error',
                                                   'Algorithm Minimum Matching Error', 'Algorithm Accuracy Metric',
                                                   'Baseline 1 Exact Matching Error',
                                                   'Baseline 1 Minimum Matching Error', 'Baseline 1 Accuracy Metric',
                                                   'Baseline 2 Exact Matching Error',
                                                   'Baseline 2 Minimum Matching Error', 'Baseline 2 Accuracy Metric',
                                                   'Baseline 3 Exact Matching Error',
                                                   'Baseline 3 Minimum Matching Error', 'Baseline 3 Accuracy Metric',
						   'Baseline 4 Minimum Matching Error', 'Baseline 4 Accuracy Metric',
                                                   'Std Algorithm Exact Matching Error',
                                                   'Std Algorithm Minimum Matching Error',
                                                   'Std Baseline 1 Exact Matching Error',
                                                   'Std Baseline 1 Minimum Matching Error',
                                                   'Std Baseline 2 Exact Matching Error',
                                                   'Std Baseline 2 Minimum Matching Error',
                                                   'Std Baseline 3 Exact Matching Error',
                                                   'Std Baseline 3 Minimum Matching Error',
						   'Std Baseline 4 Minimum Matching Error'])
    dfalg.set_index('Task Index', inplace=True)
    print(dfalg)

