path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\Doc\\Code\\ProCause"
import sys
import os
sys.path.append(path)
from src.utils.tools import save_data, load_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # For custom legend entries
import seaborn as sns
import pandas as pd
from src.utils.knn import KNNEvaluator
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, mean_absolute_error
from copy import deepcopy
import scipy.stats
from collections import defaultdict
from statistics import mean, stdev
from math import sqrt

color_map = {
    "tarnet": "#1f77b4",      # blue
    "s-learner": "#2ca02c",   # green
    "t-learner": "#ff7f0e",   # orange
    "ensemble": "#d62728",    # red
    "mlp": "#000000",         # black
    "lstm": "#7f00ff",        # dark violet
}


dash_map = {
    "mlp": "dotted",
    "lstm": "solid",
}

marker_map = {
    "mlp": "x",
    # "lstm": "o",
    "lstm": "^",
}

fixed_delta_policy = None
# fixed_delta_policy = 0.999
# fixed_delta_policy = 0.0
fixed_policy_delta_to_add = ""
if fixed_delta_policy is not None:
    fixed_policy_delta_to_add = "_fixed_" + str(fixed_delta_policy)

def get_ranking_results_per_iter(learners, intervention, num_iterations_generator, num_iterations_eval, delta=0.95, majority_vote_techniques=["borda", "kemeny", "median", "copeland", "tideman"], calculate=True):
    # goal is to get a ranking of the policies for each learner, for each evaluation and generator iteration combination --> then calculate the correlations (kendall and spearman), and eventually get the average kendall and spearman correlation for each learner
    sp_dict = {}
    kd_dict = {}

    rank_dict = {}

    true_total_perf_dict = {}

    delta_policy = delta
    if fixed_delta_policy is not None:
        delta_policy = fixed_delta_policy

    path_rank = path + "\\res\\SimBank\\ranking_" + intervention + str(delta)

    if os.path.exists(path_rank + "spearman") and not calculate:
        avg_sp_dict = load_data(path_rank + "spearman" + fixed_policy_delta_to_add)
        avg_kd_dict = load_data(path_rank + "kendall" + fixed_policy_delta_to_add)
        true_total_perf_dict = load_data(path_rank + "true_total_perf_dict" + fixed_policy_delta_to_add)
        return avg_sp_dict, avg_kd_dict, true_total_perf_dict

    true_outcome_dfs_path = "\\res\\SimBank\\online_outcome_dfs_" + intervention + str(delta_policy) + "_generator_iteration" + str(0) + "RealCause" + "['all']"
    true_outcome_dfs = load_data(path + true_outcome_dfs_path)
    
    for learner in learners + ["ensemble"]:
        sp_dict[learner + "_MLP"] = []
        sp_dict[learner + "_LSTM"] = []
        kd_dict[learner + "_MLP"] = []
        kd_dict[learner + "_LSTM"] = []

        rank_dict[learner + "_MLP"] = {}
        rank_dict[learner + "_LSTM"] = {}
        rank_dict["true"] = {}

        for iter_gen in range(num_iterations_generator):
            lstm_outcome_dfs_path = "\\res\\SimBank\\estimated_outcome_dfs_procause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + "['all']" + fixed_policy_delta_to_add
            lstm_outcome_dfs = load_data(path + lstm_outcome_dfs_path)

            mlp_outcome_dfs_path = "\\res\\SimBank\\estimated_outcome_dfs_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + "['all']" + fixed_policy_delta_to_add
            mlp_outcome_dfs = load_data(path + mlp_outcome_dfs_path)

            # true_outcome_dfs_path = "\\res\\SimBank\\online_outcome_dfs_" + intervention + str(delta_policy) + "_generator_iteration" + str(0) + "RealCause" + "['all']"
            # true_outcome_dfs = load_data(path + true_outcome_dfs_path)

            rank_dict[learner + "_LSTM"][iter_gen] = []
            rank_dict[learner + "_MLP"][iter_gen] = []
            rank_dict["true"][iter_gen] = []

            for iter_eval in range(num_iterations_eval):
                lstm_perf_dict = {}
                mlp_perf_dict = {}
                true_perf_dict = {}
                for policy in lstm_outcome_dfs:
                    if policy != 'bank':
                        lstm_perf_dict[policy] = lstm_outcome_dfs[policy][iter_eval]["outcome"].sum() / 50

                        mlp_perf_dict[policy] = mlp_outcome_dfs[policy][iter_eval]["outcome"].sum() / 50

                        # only sum the outcomes for the case_nrs that are in the estimated outcomes (so first filter the true outcomes)
                        true_outcomes = true_outcome_dfs[policy][iter_eval][true_outcome_dfs[policy][iter_eval]["case_nr"].isin(lstm_outcome_dfs[policy][iter_eval]["case_nr"].unique())]
                        true_perf_dict[policy] = true_outcomes["outcome"].sum() / 50

                        # save the estimated performances and the true performances
                        # save_data(lstm_perf_dict[policy],"estimated_performance_lstm_" + intervention + str(delta_policy) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + str(policy) + str(iter_eval))
                        # save_data(mlp_perf_dict[policy], "estimated_performance_mlp_" + intervention + str(delta_policy) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(policy) + str(iter_eval))
                        # save_data(true_perf_dict[policy], "true_performance_" + intervention + str(delta_policy) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(policy) + str(iter_eval))

                        if iter_gen == 0:
                            if policy not in true_total_perf_dict:
                                true_total_perf_dict[policy] = deepcopy(true_perf_dict[policy])
                            else:
                                true_total_perf_dict[policy] += deepcopy(true_perf_dict[policy])
                                true_total_perf_dict[policy] /= iter_eval + 1

                # rank them
                lstm_rank = sorted(lstm_perf_dict, key=lstm_perf_dict.get, reverse=True)
                mlp_rank = sorted(mlp_perf_dict, key=mlp_perf_dict.get, reverse=True)
                true_rank = sorted(true_perf_dict, key=true_perf_dict.get, reverse=True)

                rank_dict[learner + "_LSTM"][iter_gen].append(lstm_rank)
                rank_dict[learner + "_MLP"][iter_gen].append(mlp_rank)
                rank_dict["true"][iter_gen].append(true_rank)

                # save the ranking (same as below)
                save_data(data=lstm_rank, path=path + "\\res\\SimBank\\simbank_ranking_" + intervention + "_delta" + str(delta) + "_generator_iteration" + str(iter_gen) + "_LSTM_" + learner + "_eval_iteration" +  str(iter_eval) + fixed_policy_delta_to_add)
                save_data(data=mlp_rank, path=path + "\\res\\SimBank\\simbank_ranking_" + intervention + "_delta" + str(delta) + "_generator_iteration" + str(iter_gen) + "_MLP_" + learner + "_eval_iteration" +  str(iter_eval) + fixed_policy_delta_to_add)
                save_data(data=true_rank, path=path + "\\res\\SimBank\\simbank_ranking_true_" + intervention + "_delta" + str(delta) + str(iter_eval) + fixed_policy_delta_to_add)

                # calculate the spearman and kendall tau correlation between the true rank and the estimated rank
                sp_lstm, _ = scipy.stats.spearmanr(
                    [true_rank.index(policy) for policy in true_rank],
                    [lstm_rank.index(policy) for policy in true_rank]
                )

                sp_mlp, _ = scipy.stats.spearmanr(
                    [true_rank.index(policy) for policy in true_rank],
                    [mlp_rank.index(policy) for policy in true_rank]
                )

                sp_dict[learner + "_LSTM"].append([sp_lstm])
                sp_dict[learner + "_MLP"].append([sp_mlp])

                kd_lstm, _ = scipy.stats.kendalltau(
                    [true_rank.index(policy) for policy in true_rank],
                    [lstm_rank.index(policy) for policy in true_rank]
                )

                kd_mlp, _ = scipy.stats.kendalltau(
                    [true_rank.index(policy) for policy in true_rank],
                    [mlp_rank.index(policy) for policy in true_rank]
                )

                kd_dict[learner + "_LSTM"].append([kd_lstm])
                kd_dict[learner + "_MLP"].append([kd_mlp])

                # save the sp and kd
                save_data(data=sp_lstm, path=path + "\\res\\SimBank\\spearman_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + str(iter_eval) + fixed_policy_delta_to_add)
                save_data(data=sp_mlp, path=path + "\\res\\SimBank\\spearman_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(iter_eval) + fixed_policy_delta_to_add)
                save_data(data=kd_lstm, path=path + "\\res\\SimBank\\kendall_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + str(iter_eval) + fixed_policy_delta_to_add)
                save_data(data=kd_mlp, path=path + "\\res\\SimBank\\kendall_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(iter_eval) + fixed_policy_delta_to_add)

    # also get a 'majority vote' ranking
    for majority_vote in majority_vote_techniques:
        rank_dict[majority_vote + "_MLP"] = {}
        rank_dict[majority_vote + "_LSTM"] = {}

        sp_dict[majority_vote + "_MLP"] = []
        sp_dict[majority_vote + "_LSTM"] = []
        kd_dict[majority_vote + "_MLP"] = []
        kd_dict[majority_vote + "_LSTM"] = []

    for iter_gen in range(num_iterations_generator):
        for majority_vote in majority_vote_techniques:
            rank_dict[majority_vote + "_MLP"][iter_gen] = []
            rank_dict[majority_vote + "_LSTM"][iter_gen] = []

        for iter_eval in range(num_iterations_eval):
            true_rank = rank_dict["true"][iter_gen][iter_eval]

            for majority_vote in majority_vote_techniques:
                rank_mlp = calculate_majority_vote([rank_dict[learner + "_MLP"][iter_gen][iter_eval] for learner in learners], technique=majority_vote)
                rank_lstm = calculate_majority_vote([rank_dict[learner + "_LSTM"][iter_gen][iter_eval] for learner in learners], technique=majority_vote)

                rank_dict[majority_vote + "_MLP"][iter_gen].append(rank_mlp)
                rank_dict[majority_vote + "_LSTM"][iter_gen].append(rank_lstm)

               
                sp_mlp, _ = scipy.stats.spearmanr(
                    [true_rank.index(policy) for policy in true_rank],
                    [rank_mlp.index(policy) for policy in true_rank]
                )

                sp_lstm, _ = scipy.stats.spearmanr(
                    [true_rank.index(policy) for policy in true_rank],
                    [rank_lstm.index(policy) for policy in true_rank]
                )

                kd_mlp, _ = scipy.stats.kendalltau(
                    [true_rank.index(policy) for policy in true_rank],
                    [rank_mlp.index(policy) for policy in true_rank]
                )

                kd_lstm, _ = scipy.stats.kendalltau(
                    [true_rank.index(policy) for policy in true_rank],
                    [rank_lstm.index(policy) for policy in true_rank]
                )

                sp_dict[majority_vote + "_MLP"].append([sp_mlp])
                sp_dict[majority_vote + "_LSTM"].append([sp_lstm])
                kd_dict[majority_vote + "_MLP"].append([kd_mlp])
                kd_dict[majority_vote + "_LSTM"].append([kd_lstm])

                print('\n')
                print('MAJORITY VOTE: ', majority_vote)
                print("True rank: ", true_rank)
                print('rank mlp: ', rank_mlp)
                print('rank lstm: ', rank_lstm)
                print("Spearman correlation MLP: ", sp_mlp)
                print("Spearman correlation LSTM: ", sp_lstm)
                print("Kendall tau correlation MLP: ", kd_mlp)
                print("Kendall tau correlation LSTM: ", kd_lstm)

                # save the sp and kd
                save_data(data=sp_mlp, path=path + "\\res\\SimBank\\spearman_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + majority_vote + str(iter_eval) + fixed_policy_delta_to_add)
                save_data(data=sp_lstm, path=path + "\\res\\SimBank\\spearman_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + majority_vote + str(iter_eval) + fixed_policy_delta_to_add)
                save_data(data=kd_mlp, path=path + "\\res\\SimBank\\kendall_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + majority_vote + str(iter_eval) + fixed_policy_delta_to_add)
                save_data(data=kd_lstm, path=path + "\\res\\SimBank\\kendall_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + majority_vote + str(iter_eval) + fixed_policy_delta_to_add)

    # get the average spearman and kendall tau correlation for each learner
    avg_sp_dict = {}
    avg_kd_dict = {}
    for learner in learners + ["ensemble"] + majority_vote_techniques:
        avg_sp_dict[learner + "_MLP"] = np.mean([sp_dict[learner + "_MLP"][i][0] for i in range(num_iterations_generator)])
        avg_sp_dict[learner + "_LSTM"] = np.mean([sp_dict[learner + "_LSTM"][i][0] for i in range(num_iterations_generator)])

        avg_kd_dict[learner + "_MLP"] = np.mean([kd_dict[learner + "_MLP"][i][0] for i in range(num_iterations_generator)])
        avg_kd_dict[learner + "_LSTM"] = np.mean([kd_dict[learner + "_LSTM"][i][0] for i in range(num_iterations_generator)])

    # save
    save_data(avg_sp_dict, path_rank + "spearman" + fixed_policy_delta_to_add)
    save_data(avg_kd_dict, path_rank + "kendall" + fixed_policy_delta_to_add)
    save_data(true_total_perf_dict, path_rank + "true_total_perf_dict" + fixed_policy_delta_to_add)

    return avg_sp_dict, avg_kd_dict, true_total_perf_dict

def calculate_majority_vote(ranks, technique="borda"):
    if technique == "borda":
        return calculate_borda_count(ranks)
    elif technique == "kemeny":
        return calculate_kemeny_young(ranks)
    elif technique == "median":
        return calculate_median_ranking(ranks)
    elif technique == "copeland":
        return calculate_copeland_ranking(ranks)
    elif technique == "tideman":
        return calculate_tideman_ranking(ranks)
    else:
        raise ValueError("Unknown technique: {}".format(technique))

def calculate_borda_count(ranks):
    """
    Calculate the Borda count for a list of ranks.
    Each rank is a list of items in order of preference.
    """
    from collections import defaultdict

    # Initialize a score dictionary
    score = defaultdict(int)
    
    # Number of items in each rank
    n = len(ranks[0])
    
    # Iterate through each rank and assign points
    for rank in ranks:
        for i, item in enumerate(rank):
            # Highest rank gets most points: n - i
            score[item] += n - i
    
    print("ranks: ", ranks)
    print("score: ", score)

    # Sort by score descending (higher score = better rank)
    return sorted(score, key=score.get, reverse=True)


def calculate_kemeny_young(ranks):
    """
    Calculate the Kemeny-Young consensus ranking for a list of ranks.
    Each rank is a list of items in order of preference.
    """
    import itertools

    def kendall_tau_distance(rank1, rank2):
        distance = 0
        n = len(rank1)
        rank1_pos = {item: i for i, item in enumerate(rank1)}
        rank2_pos = {item: i for i, item in enumerate(rank2)}
        for i in range(n):
            for j in range(i + 1, n):
                a, b = rank1[i], rank1[j]
                if (rank2_pos[a] - rank2_pos[b]) * (rank1_pos[a] - rank1_pos[b]) < 0:
                    distance += 1
        return distance

    all_items = ranks[0]
    min_distance = float('inf')
    best_rank = None
    for perm in itertools.permutations(all_items):
        total_distance = sum(kendall_tau_distance(perm, r) for r in ranks)
        if total_distance < min_distance:
            min_distance = total_distance
            best_rank = perm
    return list(best_rank)



def calculate_median_ranking(ranks):
    """
    Calculate the median ranking for a list of ranks.
    Each rank is a list of items in order of preference.
    """
    from collections import defaultdict

    # Get all unique items
    items = ranks[0]  # Assumes all ranks have same items
    item_positions = defaultdict(list)

    # Collect the position of each item in all rankings
    for rank in ranks:
        for position, item in enumerate(rank):
            item_positions[item].append(position)

    # Compute median rank for each item
    median_ranks = {item: np.median(positions) for item, positions in item_positions.items()}

    # Sort items by median rank (lower median = higher preference)
    sorted_items = sorted(median_ranks, key=median_ranks.get)

    return sorted_items

def calculate_copeland_ranking(ranks):
    """
    Calculate the Copeland ranking for a list of rankings.
    Each rank is a list of items in order of preference.
    """
    from itertools import combinations
    from collections import defaultdict

    # Get all unique items
    items = ranks[0]  # Assuming all ranks have same items
    score = defaultdict(int)

    # Compare each pair of items
    for a, b in combinations(items, 2):
        a_wins = 0
        b_wins = 0
        for rank in ranks:
            if rank.index(a) < rank.index(b):
                a_wins += 1
            elif rank.index(b) < rank.index(a):
                b_wins += 1
        
        if a_wins > b_wins:
            score[a] += 1
            score[b] -= 1
        elif b_wins > a_wins:
            score[b] += 1
            score[a] -= 1
        # Ties result in no change

    # Sort by score descending (higher score = better rank)
    return sorted(items, key=lambda x: score[x], reverse=True)


def calculate_tideman_ranking(ranks):
    """
    Calculate the Tideman (Ranked Pairs) ranking for a list of ranks.
    """
    # Get list of all items
    items = list(ranks[0])
    n = len(items)
    index = {item: i for i, item in enumerate(items)}

    # Initialize pairwise preferences matrix
    pairwise = np.zeros((n, n), dtype=int)

    # Fill pairwise matrix
    for rank in ranks:
        for i in range(n):
            for j in range(i + 1, n):
                winner = rank[i]
                loser = rank[j]
                pairwise[index[winner]][index[loser]] += 1

    # Create list of pairs (i, j) where i beats j
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and pairwise[i][j] > pairwise[j][i]:
                strength = pairwise[i][j] - pairwise[j][i]
                pairs.append((i, j, strength))

    # Sort pairs by strength of victory descending
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Lock in pairs without creating cycles
    locked = np.zeros((n, n), dtype=bool)

    def creates_cycle(start, end, graph):
        """Detect if adding an edge would create a cycle using DFS."""
        if start == end:
            return True
        for i in range(n):
            if graph[end][i]:
                if creates_cycle(start, i, graph):
                    return True
        return False

    for winner, loser, _ in pairs:
        if not creates_cycle(winner, loser, locked):
            locked[winner][loser] = True

    # Determine ranking from the locked graph
    incoming_edges = np.sum(locked, axis=0)
    ranking = []
    used = set()

    while len(ranking) < n:
        for i in range(n):
            if incoming_edges[i] == 0 and i not in used:
                ranking.append(items[i])
                used.add(i)
                # Remove outgoing edges from this node
                for j in range(n):
                    if locked[i][j]:
                        incoming_edges[j] -= 1
                break

    return ranking

def plot_delta_ranking_results_per_iter(avg_sp_dict_list, avg_kd_dict_list, delta_list, majority_vote_techniques=["borda", "kemeny", "median", "copeland", "tideman"]):
    # Set a color cycle with enough distinct colors
    # colors = plt.cm.tab20(np.linspace(0, 1, len(avg_sp_dict_list[0])))

    # make sure the avg_sp_dict and avg_kd_dict are in this order: first everything with _MLP, then everything with _LSTM, then ensemmbles (with ensemble LSTM first) and majority vote techniques
    
    # Plot for Spearman Correlation
    plt.figure(figsize=(12, 6))
    for idx, learner in enumerate(avg_sp_dict_list[0]):
        sp_list = []

        # check if learner in majority_vote_techniques and if so (note that is has _MLP or _LSTM attached, so need to remove that)
        learner_name = learner.split("_")[0]
        if learner_name in majority_vote_techniques:
            continue

        for i in range(len(avg_sp_dict_list)):
            sp_list.append(avg_sp_dict_list[i][learner])
        
        marker = 'x' if 'MLP' in learner else 'o' if 'LSTM' in learner else '.'
        print(color_map.get(learner.lower(), None))
        print(learner.lower())
        plt.plot(delta_list, sp_list, label=learner, marker=marker, color=color_map.get(learner.lower(), None))
    
    plt.xlabel("Delta")
    plt.ylabel("Spearman Correlation")
    plt.title("Ranking Results per Iteration")
    plt.grid()
    
    # Move legend outside the plot to the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()
    
    # Plot for Kendall Tau Correlation
    plt.figure(figsize=(12, 6))
    for idx, learner in enumerate(avg_kd_dict_list[0]):
        kd_list = []

        # check if learner in majority_vote_techniques and if so (note that is has _MLP or _LSTM attached, so need to remove that)
        learner_name = learner.split("_")[0]
        if learner_name in majority_vote_techniques:
            continue

        for i in range(len(avg_kd_dict_list)):
            kd_list.append(avg_kd_dict_list[i][learner])
        
        marker = 'x' if 'MLP' in learner else 'o' if 'LSTM' in learner else '.'
        plt.plot(delta_list, kd_list, label=learner, marker=marker, color=color_map.get(learner.lower(), None))
    
    plt.xlabel("Delta")
    plt.ylabel("Kendall Tau Correlation")
    plt.title("Ranking Results per Iteration")
    plt.grid()
    
    # Move legend outside the plot to the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()


def plot_CE_performance(avg_true_dict_list, delta_list):
    # plot a line plot for each policy over the delta values
    plt.figure(figsize=(12, 6))
    plt.title("CE Performance")
    plt.xlabel("Delta")
    plt.ylabel("PEHE")

    print(avg_true_dict_list)
    # avg_true_mae_dict_list is a list of dictionaries (going from low to high delta), where the dictionary has as keys the policies and as values the average MAE for that policy

    # so we need to plot the values for each policy over the delta values
    for policy in avg_true_dict_list[0]:
        values = [avg_true_mae_dict[policy] for avg_true_mae_dict in avg_true_dict_list]
        plt.plot(delta_list, values, label=policy, marker='o', markersize=8)

    plt.legend()
    plt.grid()
    plt.show()

def calculate_wssd_ensembles(learners, intervention, num_iterations_generator, num_iterations_eval, policies=["all"], delta=0.95):
    # import wasserstein distance
    from scipy.stats import wasserstein_distance as wssd
    from copy import deepcopy

    ensemble_pc_outcome_dfs = []
    ensemble_rc_outcome_dfs = []

    delta_policy = delta
    if fixed_delta_policy is not None:
        delta_policy = fixed_delta_policy

    true_outcome_dfs_path = "\\res\\SimBank\\online_outcome_dfs_" + intervention + str(delta_policy) + "_generator_iteration" + str(0) + "RealCause" + "['all']"
    # check if the file exists, if not then it should be with 'all' in the name
    if not os.path.exists(path + true_outcome_dfs_path):
        true_outcome_dfs_path = "\\res\\SimBank\\online_outcome_dfs_" + intervention + str(delta_policy) + "_generator_iteration" + str(0) + "RealCause" + str(policies)
    true_outcome_dfs = load_data(path + true_outcome_dfs_path)

    ensemble_wssd_pc = []
    ensemble_wssd_rc = []

    # also get a full wssd per case_nr for ensemble, tarnet, s-learner and t-learner for rc and pc, but only for random policy (get a list of wssd values for each case_nr, but a mean over the iter_gen and iter_eval)
    ensemble_random_wssd_rc = []
    ensemble_random_wssd_pc = []
    tarnet_random_wssd_rc = []
    tarnet_random_wssd_pc = []
    slearner_random_wssd_rc = []
    slearner_random_wssd_pc = []
    tlearner_random_wssd_rc = []
    tlearner_random_wssd_pc = []

    for iter_gen in range(num_iterations_generator):
        print("Iteration generator ", iter_gen)
        dict_to_add_pc = {}
        dict_to_add_rc = {}
        wssd_dict_pc = {}
        wssd_dict_rc = {}

        ensemble_procause_outcome_dict = {}
        ensemble_realcause_outcome_dict = {}
        for learner in learners:
            path_pc = "\\res\\SimBank\\estimated_outcome_dfs_procause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + "['all']" + fixed_policy_delta_to_add
            path_rc = "\\res\\SimBank\\estimated_outcome_dfs_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + "['all']" + fixed_policy_delta_to_add
            if not os.path.exists(path + path_pc):
                path_pc = "\\res\\SimBank\\estimated_outcome_dfs_procause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + str(policies) + fixed_policy_delta_to_add
            if not os.path.exists(path + path_rc):
                path_rc = "\\res\\SimBank\\estimated_outcome_dfs_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(policies) + fixed_policy_delta_to_add
            ensemble_procause_outcome_dict[learner] = load_data(path + path_pc)
            ensemble_realcause_outcome_dict[learner] = load_data(path + path_rc)

            other_est_path = "c:\\Users\\u0166838\\OneDrive - KU Leuven\\Documents\\SimBank\\estimated_outcome_dfs_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(policies) + fixed_policy_delta_to_add
            lol = load_data(other_est_path)


        for policy in ensemble_procause_outcome_dict["TarNet"]:
            # if policy == "bank":
            #     continue
            dict_to_add_pc[policy] = []
            dict_to_add_rc[policy] = []
            wssd_dict_pc[policy] = []
            wssd_dict_rc[policy] = []

            for iter_eval in range(num_iterations_eval):
                if iter_eval >= len(ensemble_procause_outcome_dict["TarNet"][policy]):
                    continue
                print("Policy ", policy, " iteration eval ", iter_eval)

                df_pc = deepcopy(ensemble_procause_outcome_dict["TarNet"][policy][iter_eval])
                df_pc["outcome"] += deepcopy(ensemble_procause_outcome_dict["S-Learner"][policy][iter_eval]["outcome"]) + deepcopy(ensemble_procause_outcome_dict["T-Learner"][policy][iter_eval]["outcome"])
                df_pc["outcome"] = df_pc["outcome"] / 3
                dict_to_add_pc[policy].append(df_pc)

                df_rc = deepcopy(ensemble_realcause_outcome_dict["TarNet"][policy][iter_eval])
                df_rc["outcome"] += deepcopy(ensemble_realcause_outcome_dict["S-Learner"][policy][iter_eval]["outcome"]) + deepcopy(ensemble_realcause_outcome_dict["T-Learner"][policy][iter_eval]["outcome"])
                df_rc["outcome"] = df_rc["outcome"] / 3
                dict_to_add_rc[policy].append(df_rc)

                if policy == "bank":
                    continue

                wssd_list_pc = []
                wssd_list_rc = []

                wssd_list_tarnet_pc = []
                wssd_list_tarnet_rc = []
                wssd_list_slearner_pc = []
                wssd_list_slearner_rc = []
                wssd_list_tlearner_pc = []
                wssd_list_tlearner_rc = []

                for case_nr in true_outcome_dfs[policy][iter_eval]["case_nr"].unique():
                    if case_nr not in df_rc["case_nr"].unique():
                        wssd_list_pc.append(0)
                        wssd_list_rc.append(0)

                        if policy == "random" and iter_eval == 0 and iter_gen == 0:
                            wssd_list_tarnet_pc.append(0)
                            wssd_list_tarnet_rc.append(0)
                            wssd_list_slearner_pc.append(0)
                            wssd_list_slearner_rc.append(0)
                            wssd_list_tlearner_pc.append(0)
                            wssd_list_tlearner_rc.append(0)

                        continue
                    true_outcomes = true_outcome_dfs[policy][iter_eval][true_outcome_dfs[policy][iter_eval]["case_nr"] == case_nr]
                    estimated_outcomes_rc = df_rc[df_rc["case_nr"] == case_nr]
                    estimated_outcomes_pc = df_pc[df_pc["case_nr"] == case_nr]
                    wssd_pc = wssd(true_outcomes["outcome"], estimated_outcomes_pc["outcome"])
                    wssd_rc = wssd(true_outcomes["outcome"], estimated_outcomes_rc["outcome"])

                    print(wssd_pc)
                    wssd_list_pc.append(wssd_pc)
                    wssd_list_rc.append(wssd_rc)


                    if policy == "random":
                        wssd_tarnet_pc = wssd(true_outcomes["outcome"], ensemble_procause_outcome_dict["TarNet"]["random"][iter_eval][ensemble_procause_outcome_dict["TarNet"]["random"][iter_eval]["case_nr"] == case_nr]["outcome"])
                        wssd_tarnet_rc = wssd(true_outcomes["outcome"], ensemble_realcause_outcome_dict["TarNet"]["random"][iter_eval][ensemble_realcause_outcome_dict["TarNet"]["random"][iter_eval]["case_nr"] == case_nr]["outcome"])
                        wssd_slearner_pc = wssd(true_outcomes["outcome"], ensemble_procause_outcome_dict["S-Learner"]["random"][iter_eval][ensemble_procause_outcome_dict["S-Learner"]["random"][iter_eval]["case_nr"] == case_nr]["outcome"])
                        wssd_slearner_rc = wssd(true_outcomes["outcome"], ensemble_realcause_outcome_dict["S-Learner"]["random"][iter_eval][ensemble_realcause_outcome_dict["S-Learner"]["random"][iter_eval]["case_nr"] == case_nr]["outcome"])
                        wssd_tlearner_pc = wssd(true_outcomes["outcome"], ensemble_procause_outcome_dict["T-Learner"]["random"][iter_eval][ensemble_procause_outcome_dict["T-Learner"]["random"][iter_eval]["case_nr"] == case_nr]["outcome"])
                        wssd_tlearner_rc = wssd(true_outcomes["outcome"], ensemble_realcause_outcome_dict["T-Learner"]["random"][iter_eval][ensemble_realcause_outcome_dict["T-Learner"]["random"][iter_eval]["case_nr"] == case_nr]["outcome"])
                        
                        # check if the length of the list is larger than the index of case_nr in unique, if so, add the wssd to the existing value, otherwise append
                        case_nr_index = df_rc["case_nr"].unique().tolist().index(case_nr)
                        if len(tarnet_random_wssd_pc) > case_nr_index:
                            tarnet_random_wssd_pc[case_nr_index] += wssd_tarnet_pc
                            tarnet_random_wssd_rc[case_nr_index] += wssd_tarnet_rc
                            slearner_random_wssd_pc[case_nr_index] += wssd_slearner_pc
                            slearner_random_wssd_rc[case_nr_index] += wssd_slearner_rc
                            tlearner_random_wssd_pc[case_nr_index] += wssd_tlearner_pc
                            tlearner_random_wssd_rc[case_nr_index] += wssd_tlearner_rc
                        else:
                            tarnet_random_wssd_pc.append(wssd_tarnet_pc)
                            tarnet_random_wssd_rc.append(wssd_tarnet_rc)
                            slearner_random_wssd_pc.append(wssd_slearner_pc)
                            slearner_random_wssd_rc.append(wssd_slearner_rc)
                            tlearner_random_wssd_pc.append(wssd_tlearner_pc)
                            tlearner_random_wssd_rc.append(wssd_tlearner_rc)

                if policy == "random":
                    if len(ensemble_random_wssd_pc) > 0:
                        ensemble_random_wssd_pc = [ensemble_random_wssd_pc[i] + wssd_list_pc[i] for i in range(len(wssd_list_pc))]
                        ensemble_random_wssd_rc = [ensemble_random_wssd_rc[i] + wssd_list_rc[i] for i in range(len(wssd_list_rc))]
                    else:
                        ensemble_random_wssd_pc = wssd_list_pc
                        ensemble_random_wssd_rc = wssd_list_rc

                wssd_dict_pc[policy].append(sum(wssd_list_pc) / len(wssd_list_pc))
                wssd_dict_rc[policy].append(sum(wssd_list_rc) / len(wssd_list_rc))

                # save the wssd distcance for the ensemble per policy, per iter_eval, per iter_gen
                save_data(path=path + "\\res\\SimBank\\wssd_procause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + policy + str(iter_eval) + fixed_policy_delta_to_add + ".pkl", data=sum(wssd_list_pc) / len(wssd_list_pc))
                save_data(path=path + "\\res\\SimBank\\wssd_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + policy + str(iter_eval) + fixed_policy_delta_to_add + ".pkl", data=sum(wssd_list_rc) / len(wssd_list_rc))

        ensemble_pc_outcome_dfs.append(dict_to_add_pc)
        ensemble_rc_outcome_dfs.append(dict_to_add_rc)
        # save the outcome_dfs
        path_pc_ensemble = "\\res\\SimBank\\estimated_outcome_dfs_procause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + "ensemble" + str(policies) + fixed_policy_delta_to_add
        path_rc_ensemble = "\\res\\SimBank\\estimated_outcome_dfs_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + "ensemble" + str(policies) + fixed_policy_delta_to_add
        save_data(path=path + path_pc_ensemble, data=ensemble_pc_outcome_dfs[iter_gen])
        save_data(path=path + path_rc_ensemble, data=ensemble_rc_outcome_dfs[iter_gen])
        ensemble_wssd_pc.append(wssd_dict_pc)
        ensemble_wssd_rc.append(wssd_dict_rc)
        print('\n')

    ensemble_avg_wssd_pc = {}
    ensemble_avg_wssd_rc = {}
    ensemble_conf_int_pc = {}
    ensemble_conf_int_rc = {}
    for policy in ensemble_wssd_pc[0]:

        ensemble_avg_wssd_pc[policy] = np.mean([ensemble_wssd_pc[i][policy] for i in range(num_iterations_generator)])
        ensemble_avg_wssd_rc[policy] = np.mean([ensemble_wssd_rc[i][policy] for i in range(num_iterations_generator)])
        ensemble_conf_int_pc[policy] = np.std([ensemble_wssd_pc[i][policy] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator) * 1.96
        ensemble_conf_int_rc[policy] = np.std([ensemble_wssd_rc[i][policy] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator) * 1.96

    print("Ensemble WSSD ProCause: ", ensemble_avg_wssd_pc)
    print("Ensemble WSSD RealCause: ", ensemble_avg_wssd_rc)

    # save the results
    save_data(path=path + "\\res\\SimBank\\ensemble_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=ensemble_avg_wssd_pc)
    save_data(path=path + "\\res\\SimBank\\ensemble_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=ensemble_avg_wssd_rc)
    save_data(path=path + "\\res\\SimBank\\ensemble_conf_int_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=ensemble_conf_int_pc)
    save_data(path=path + "\\res\\SimBank\\ensemble_conf_int_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=ensemble_conf_int_rc)

    # grab the average wssd for the ensemble, tarnet, s-learner and t-learner for the random policy over all iter_gen and iter_eval
    ensemble_random_wssd_pc = [ensemble_random_wssd_pc[i] / num_iterations_generator for i in range(len(ensemble_random_wssd_pc))]
    ensemble_random_wssd_rc = [ensemble_random_wssd_rc[i] / num_iterations_generator for i in range(len(ensemble_random_wssd_rc))]
    tarnet_random_wssd_pc = [tarnet_random_wssd_pc[i] / num_iterations_generator for i in range(len(tarnet_random_wssd_pc))]
    tarnet_random_wssd_rc = [tarnet_random_wssd_rc[i] / num_iterations_generator for i in range(len(tarnet_random_wssd_rc))]
    slearner_random_wssd_pc = [slearner_random_wssd_pc[i] / num_iterations_generator for i in range(len(slearner_random_wssd_pc))]
    slearner_random_wssd_rc = [slearner_random_wssd_rc[i] / num_iterations_generator for i in range(len(slearner_random_wssd_rc))]
    tlearner_random_wssd_pc = [tlearner_random_wssd_pc[i] / num_iterations_generator for i in range(len(tlearner_random_wssd_pc))]
    tlearner_random_wssd_rc = [tlearner_random_wssd_rc[i] / num_iterations_generator for i in range(len(tlearner_random_wssd_rc))]

    # also over the iter_eval
    ensemble_random_wssd_pc = [ensemble_random_wssd_pc[i] / num_iterations_eval for i in range(len(ensemble_random_wssd_pc))]
    ensemble_random_wssd_rc = [ensemble_random_wssd_rc[i] / num_iterations_eval for i in range(len(ensemble_random_wssd_rc))]
    tarnet_random_wssd_pc = [tarnet_random_wssd_pc[i] / num_iterations_eval for i in range(len(tarnet_random_wssd_pc))]
    tarnet_random_wssd_rc = [tarnet_random_wssd_rc[i] / num_iterations_eval for i in range(len(tarnet_random_wssd_rc))]
    slearner_random_wssd_pc = [slearner_random_wssd_pc[i] / num_iterations_eval for i in range(len(slearner_random_wssd_pc))]
    slearner_random_wssd_rc = [slearner_random_wssd_rc[i] / num_iterations_eval for i in range(len(slearner_random_wssd_rc))]
    tlearner_random_wssd_pc = [tlearner_random_wssd_pc[i] / num_iterations_eval for i in range(len(tlearner_random_wssd_pc))]
    tlearner_random_wssd_rc = [tlearner_random_wssd_rc[i] / num_iterations_eval for i in range(len(tlearner_random_wssd_rc))]

    save_data(path=path + "\\res\\SimBank\\ensemble_random_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=ensemble_random_wssd_pc)
    save_data(path=path + "\\res\\SimBank\\ensemble_random_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=ensemble_random_wssd_rc)
    save_data(path=path + "\\res\\SimBank\\tarnet_random_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=tarnet_random_wssd_pc)
    save_data(path=path + "\\res\\SimBank\\tarnet_random_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=tarnet_random_wssd_rc)
    save_data(path=path + "\\res\\SimBank\\slearner_random_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=slearner_random_wssd_pc)
    save_data(path=path + "\\res\\SimBank\\slearner_random_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=slearner_random_wssd_rc)
    save_data(path=path + "\\res\\SimBank\\tlearner_random_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=tlearner_random_wssd_pc)
    save_data(path=path + "\\res\\SimBank\\tlearner_random_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=tlearner_random_wssd_rc)

    return ensemble_avg_wssd_pc, ensemble_avg_wssd_rc, ensemble_conf_int_pc, ensemble_conf_int_rc


def get_wssd_results(intervention, num_iterations_generator, num_iterations_eval, policies=["all"], delta=0.95, calculate=True, calculate_raw=False):

    if "all" in policies:
        policies = ["random", "S-Learner_LSTM", "T-Learner_LSTM", "TarNet_LSTM", "S-Learner_Vanilla_NN", "T-Learner_Vanilla_NN", "TarNet_Vanilla_NN"]
        policy_name = "['all']"
    else:
        policy_name = policies

    # check if ensemble results are already calculated
    if os.path.exists(path + "\\res\\SimBank\\ensemble_wssd_pc" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl") and not calculate_raw:
        ensemble_avg_wssd_pc = load_data(path + "\\res\\SimBank\\ensemble_wssd_pc" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl")
        ensemble_avg_wssd_rc = load_data(path + "\\res\\SimBank\\ensemble_wssd_rc" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl")
        ensemble_conf_int_pc = load_data(path + "\\res\\SimBank\\ensemble_conf_int_pc" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl")
        ensemble_conf_int_rc = load_data(path + "\\res\\SimBank\\ensemble_conf_int_rc" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl")
    else:
        ensemble_avg_wssd_pc, ensemble_avg_wssd_rc, ensemble_conf_int_pc, ensemble_conf_int_rc = calculate_wssd_ensembles(["TarNet", "S-Learner", "T-Learner"], intervention, num_iterations_generator, num_iterations_eval, policies=policy_name, delta=delta)
    # ensemble_avg_wssd_pc, ensemble_avg_wssd_rc, ensemble_conf_int_pc, ensemble_conf_int_rc = calculate_wssd_ensembles(["TarNet", "S-Learner", "T-Learner"], intervention, num_iterations_generator, num_iterations_eval, policies=policy_name, delta=delta)


    if os.path.exists(path + "\\res\\SimBank\\full_dict_wssd" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl") and not calculate:
        full_dict_wssd = load_data(path + "\\res\\SimBank\\full_dict_wssd" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl")
        full_dict_conf_int = load_data(path + "\\res\\SimBank\\full_dict_conf_int" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl")
        return full_dict_wssd, full_dict_conf_int

    print("INTERVENTION:", intervention, "\n")
    realcause_avg_TarNet = {}
    realcause_avg_S_Learner = {}
    realcause_avg_T_Learner = {}
    procause_avg_TarNet = {}
    procause_avg_S_Learner = {}
    procause_avg_T_Learner = {}

    for policy in policies:
        if policy == "bank":
            continue
        print("Policy:", policy)
        realcause_avg_TarNet[policy] = {}
        realcause_avg_S_Learner[policy] = {}
        realcause_avg_T_Learner[policy] = {}
        procause_avg_TarNet[policy] = {}
        procause_avg_S_Learner[policy] = {}
        procause_avg_T_Learner[policy] = {}
        for iter_gen in range(num_iterations_generator):
            for iter_eval in range(num_iterations_eval):
                path1 = path + "\\res\\SimBank\\wssd_realcause_eval_iteration" + str(iter_eval) + "_" + intervention + "TarNet" + str(delta) +"_" + policy + "_generator_iteration" + str(iter_gen) + fixed_policy_delta_to_add
                data1 = load_data(path1)
                # print("RealCause for TarNet:", data1)
                if iter_gen in realcause_avg_TarNet[policy]:
                    realcause_avg_TarNet[policy][iter_gen] += data1
                else:
                    realcause_avg_TarNet[policy][iter_gen] = data1
                
                path2 = path + "\\res\\SimBank\\wssd_procause_eval_iteration" + str(iter_eval) + "_" + intervention + "TarNet" + str(delta) +"_" + policy + "_generator_iteration" + str(iter_gen) + fixed_policy_delta_to_add
                data2 = load_data(path2)
                # print("ProCause for TarNet:", data2)
                if iter_gen in procause_avg_TarNet[policy]:
                    procause_avg_TarNet[policy][iter_gen] += data2
                else:
                    procause_avg_TarNet[policy][iter_gen] = data2

                path3 = path + "\\res\\SimBank\\wssd_procause_eval_iteration" + str(iter_eval) + "_" + intervention + "S-Learner" + str(delta) +"_" + policy + "_generator_iteration" + str(iter_gen) + fixed_policy_delta_to_add
                data3 = load_data(path3)
                # print("ProCause for S-Learner:", data3)
                if iter_gen in procause_avg_S_Learner[policy]:
                    procause_avg_S_Learner[policy][iter_gen] += data3
                else:
                    procause_avg_S_Learner[policy][iter_gen] = data3

                path4 = path + "\\res\\SimBank\\wssd_realcause_eval_iteration" + str(iter_eval) + "_" + intervention + "S-Learner" + str(delta) +"_" + policy + "_generator_iteration" + str(iter_gen) + fixed_policy_delta_to_add
                data4 = load_data(path4)
                # print("RealCause for S-Learner:", data4)
                if iter_gen in realcause_avg_S_Learner[policy]:
                    realcause_avg_S_Learner[policy][iter_gen] += data4
                else:
                    realcause_avg_S_Learner[policy][iter_gen] = data4

                path5 = path + "\\res\\SimBank\\wssd_procause_eval_iteration" + str(iter_eval) + "_" + intervention + "T-Learner" + str(delta) +"_" + policy + "_generator_iteration" + str(iter_gen) + fixed_policy_delta_to_add
                data5 = load_data(path5)
                # print("ProCause for T-Learner:", data5)
                if iter_gen in procause_avg_T_Learner[policy]:
                    procause_avg_T_Learner[policy][iter_gen] += data5
                else:
                    procause_avg_T_Learner[policy][iter_gen] = data5

                
                path6 = path + "\\res\\SimBank\\wssd_realcause_eval_iteration" + str(iter_eval) + "_" + intervention + "T-Learner" + str(delta) +"_" + policy + "_generator_iteration" + str(iter_gen) + fixed_policy_delta_to_add
                data6 = load_data(path6)
                # print("RealCause for T-Learner:", data6)
                if iter_gen in realcause_avg_T_Learner[policy]:
                    realcause_avg_T_Learner[policy][iter_gen] += data6
                else:
                    realcause_avg_T_Learner[policy][iter_gen] = data6

            realcause_avg_TarNet[policy][iter_gen] = realcause_avg_TarNet[policy][iter_gen] / num_iterations_eval
            realcause_avg_S_Learner[policy][iter_gen] = realcause_avg_S_Learner[policy][iter_gen] / num_iterations_eval
            realcause_avg_T_Learner[policy][iter_gen] = realcause_avg_T_Learner[policy][iter_gen] / num_iterations_eval
            procause_avg_TarNet[policy][iter_gen] = procause_avg_TarNet[policy][iter_gen] / num_iterations_eval
            procause_avg_S_Learner[policy][iter_gen] = procause_avg_S_Learner[policy][iter_gen] / num_iterations_eval
            procause_avg_T_Learner[policy][iter_gen] = procause_avg_T_Learner[policy][iter_gen] / num_iterations_eval

    print("\n")
    print("RealCause for TarNet:", realcause_avg_TarNet)
    print("RealCause for S-Learner:", realcause_avg_S_Learner)
    print("RealCause for T-Learner:", realcause_avg_T_Learner)
    print("ProCause for TarNet:", procause_avg_TarNet)
    print("ProCause for S-Learner:", procause_avg_S_Learner)
    print("ProCause for T-Learner:", procause_avg_T_Learner)

    # now get the average over the generator iterations for each policy
    total_realcause_avg_TarNet = {}
    total_realcause_avg_S_Learner = {}
    total_realcause_avg_T_Learner = {}
    total_procause_avg_TarNet = {}
    total_procause_avg_S_Learner = {}
    total_procause_avg_T_Learner = {}

    conf_int_realcause_TarNet = {}
    conf_int_realcause_S_Learner = {}
    conf_int_realcause_T_Learner = {}
    conf_int_procause_TarNet = {}
    conf_int_procause_S_Learner = {}
    conf_int_procause_T_Learner = {}

    for policy in policies:
        if policy == "bank":
            continue
        # total_realcause_avg[policy] = np.mean([realcause_avg[policy][i] for i in range(num_iterations_generator)])
        total_realcause_avg_TarNet[policy] = np.mean([realcause_avg_TarNet[policy][i] for i in range(num_iterations_generator)])
        total_realcause_avg_S_Learner[policy] = np.mean([realcause_avg_S_Learner[policy][i] for i in range(num_iterations_generator)])
        total_realcause_avg_T_Learner[policy] = np.mean([realcause_avg_T_Learner[policy][i] for i in range(num_iterations_generator)])
        total_procause_avg_TarNet[policy] = np.mean([procause_avg_TarNet[policy][i] for i in range(num_iterations_generator)])
        total_procause_avg_S_Learner[policy] = np.mean([procause_avg_S_Learner[policy][i] for i in range(num_iterations_generator)])
        total_procause_avg_T_Learner[policy] = np.mean([procause_avg_T_Learner[policy][i] for i in range(num_iterations_generator)])

        # conf_int_realcause[policy] = np.std([realcause_avg[policy][i] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator)
        conf_int_realcause_TarNet[policy] = np.std([realcause_avg_TarNet[policy][i] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator) * 1.96
        conf_int_realcause_S_Learner[policy] = np.std([realcause_avg_S_Learner[policy][i] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator) * 1.96
        conf_int_realcause_T_Learner[policy] = np.std([realcause_avg_T_Learner[policy][i] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator) * 1.96
        conf_int_procause_TarNet[policy] = np.std([procause_avg_TarNet[policy][i] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator) * 1.96
        conf_int_procause_S_Learner[policy] = np.std([procause_avg_S_Learner[policy][i] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator) * 1.96
        conf_int_procause_T_Learner[policy] = np.std([procause_avg_T_Learner[policy][i] for i in range(num_iterations_generator)]) / np.sqrt(num_iterations_generator) * 1.96

    print("\n")
    # print("Total RealCause:", total_realcause_avg)
    print("Total RealCause for TarNet:", total_realcause_avg_TarNet)
    print("Total RealCause for S-Learner:", total_realcause_avg_S_Learner)
    print("Total RealCause for T-Learner:", total_realcause_avg_T_Learner)
    print("Total ProCause for TarNet:", total_procause_avg_TarNet)
    print("Total ProCause for S-Learner:", total_procause_avg_S_Learner)
    print("Total ProCause for T-Learner:", total_procause_avg_T_Learner)
    print("Ttotal ProCause Ensemble:", ensemble_avg_wssd_pc)
    print("Total RealCause Ensemble:", ensemble_avg_wssd_rc)

    print("\n")
    # print("Confidence interval RealCause:", conf_int_realcause)
    print("Confidence interval RealCause for TarNet:", conf_int_realcause_TarNet)
    print("Confidence interval RealCause for S-Learner:", conf_int_realcause_S_Learner)
    print("Confidence interval RealCause for T-Learner:", conf_int_realcause_T_Learner)
    print("Confidence interval ProCause for TarNet:", conf_int_procause_TarNet)
    print("Confidence interval ProCause for S-Learner:", conf_int_procause_S_Learner)
    print("Confidence interval ProCause for T-Learner:", conf_int_procause_T_Learner)
    print("Confidence interval ProCause Ensemble:", ensemble_conf_int_pc)
    print("Confidence interval RealCause Ensemble:", ensemble_conf_int_rc)

    full_dict_wssd = {"TarNet-MLP": total_realcause_avg_TarNet, "S-Learner-MLP": total_realcause_avg_S_Learner, "T-Learner-MLP": total_realcause_avg_T_Learner, "TarNet-LSTM": total_procause_avg_TarNet, "S-Learner-LSTM": total_procause_avg_S_Learner, "T-Learner-LSTM": total_procause_avg_T_Learner, "Ensemble ProCause": ensemble_avg_wssd_pc, "Ensemble RealCause": ensemble_avg_wssd_rc}
    full_dict_conf_int = {"TarNet-MLP": conf_int_realcause_TarNet, "S-Learner-MLP": conf_int_realcause_S_Learner, "T-Learner-MLP": conf_int_realcause_T_Learner, "TarNet-LSTM": conf_int_procause_TarNet, "S-Learner-LSTM": conf_int_procause_S_Learner, "T-Learner-LSTM": conf_int_procause_T_Learner, "Ensemble ProCause": ensemble_conf_int_pc, "Ensemble RealCause": ensemble_conf_int_rc}


    # save the results
    save_data(path=path + "\\res\\SimBank\\full_dict_wssd" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl", data=full_dict_wssd)
    save_data(path=path + "\\res\\SimBank\\full_dict_conf_int" + intervention + str(delta) + str(policy_name) + fixed_policy_delta_to_add + ".pkl", data=full_dict_conf_int)

    return full_dict_wssd, full_dict_conf_int


def get_single_wssd_result(intervention, iter_gen, iter_eval, policy, delta, learner, model_type):
    if learner == "ensemble":
        if model_type == "mlp":
            path_wssd = path + "\\res\\SimBank\\wssd_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + policy + str(iter_eval) + fixed_policy_delta_to_add + ".pkl"
        else:
            path_wssd = path + "\\res\\SimBank\\wssd_procause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + policy + str(iter_eval) + fixed_policy_delta_to_add + ".pkl"
    else:
        if model_type == "mlp":
            path_wssd = path + "\\res\\SimBank\\wssd_realcause_eval_iteration" + str(iter_eval) + "_" + intervention + learner + str(delta) + "_" + policy + "_generator_iteration" + str(iter_gen) + fixed_policy_delta_to_add
        else:
            path_wssd = path + "\\res\\SimBank\\wssd_procause_eval_iteration" + str(iter_eval) + "_" + intervention + learner + str(delta) + "_" + policy + "_generator_iteration" + str(iter_gen) + fixed_policy_delta_to_add

    wssd_result = load_data(path_wssd)

    return wssd_result

def get_single_ranking_result_sp(intervention, iter_gen, iter_eval, delta, learner, model_type):
    if model_type == "mlp":
        path_sp = path + "\\res\\SimBank\\spearman_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(iter_eval) + fixed_policy_delta_to_add
    else:
        path_sp = path + "\\res\\SimBank\\spearman_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + str(iter_eval) + fixed_policy_delta_to_add

    sp_result = load_data(path_sp)

    return sp_result

def get_single_ranking_result_ks(intervention, iter_gen, iter_eval, delta, learner, model_type):
    if model_type == "mlp":
        path_ks = path + "\\res\\SimBank\\kendall_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(iter_eval) + fixed_policy_delta_to_add
    else:
        path_ks = path + "\\res\\SimBank\\kendall_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + str(iter_eval) + fixed_policy_delta_to_add

    ks_result = load_data(path_ks)

    return ks_result

def plot_actual_delta_results(metric, ax, intervention, num_iterations_generator, num_iterations_eval, delta_list, policies=["all"], calculate=True, plot_each_policy=False, to_plot="model_stats_non_random", single=False):
    # LOAD ALL THE RESULTS, and aggregate in the different ways (per learner and model_type combined, then per learner, then per model_type, and every time for random policy and for non-random policies)
    learners = ["TarNet", "S-Learner", "T-Learner", "ensemble"]
    majority_votes = ["borda", "kemeny", "median", "copeland", "tideman"]
    if metric != "wssd":
        learners = learners + majority_votes
    model_types = ["mlp", "lstm"]
    if "all" in policies:
        policies = ["random", "S-Learner_LSTM", "T-Learner_LSTM", "TarNet_LSTM", "S-Learner_Vanilla_NN", "T-Learner_Vanilla_NN", "TarNet_Vanilla_NN"]

    if calculate:
        # Helper function to compute mean and standard error
        def compute_avg_se(values):
            n = len(values)
            if n == 0:
                return (None, None)
            elif n == 1:
                return (values[0], 0.0)
            else:
                return (mean(values), stdev(values) / sqrt(n))

        # Data collectors
        results_by_learner_model_non_random = defaultdict(lambda: defaultdict(list))
        results_by_learner_non_random = defaultdict(lambda: defaultdict(list))
        results_by_model_non_random = defaultdict(lambda: defaultdict(list))

        results_by_learner_model_random = defaultdict(lambda: defaultdict(list))
        results_by_learner_random = defaultdict(lambda: defaultdict(list))
        results_by_model_random = defaultdict(lambda: defaultdict(list))

        results_by_learner_model_per_policy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        # Main loop
        for learner in learners:
            for model_type in model_types:
                for delta in delta_list:
                    for policy in policies:
                        # just skip because it is a ranking of the policies, so one pass is enough
                        if metric != "wssd" and policy != "random":
                            continue

                        for iter_gen in range(num_iterations_generator):
                            for iter_eval in range(num_iterations_eval):

                                if metric == "wssd":
                                    res = get_single_wssd_result(
                                        intervention, iter_gen, iter_eval, policy, delta, learner, model_type
                                    )
                                    print(res)
                                    key_lm = f"{learner}_{model_type}"
                                elif metric == "spearman":
                                    res = get_single_ranking_result_sp(
                                        intervention, iter_gen, iter_eval, delta, learner, model_type
                                    )
                                    key_lm = f"{learner}_{model_type}"
                                else:
                                    res = get_single_ranking_result_ks(
                                        intervention, iter_gen, iter_eval, delta, learner, model_type
                                    )
                                    key_lm = f"{learner}_{model_type}"

                                if policy == "random":
                                    results_by_learner_model_random[key_lm][delta].append(res)
                                    results_by_learner_random[learner][delta].append(res)
                                    results_by_model_random[model_type][delta].append(res)
                                else:
                                    results_by_learner_model_non_random[key_lm][delta].append(res)
                                    results_by_learner_non_random[learner][delta].append(res)
                                    results_by_model_non_random[model_type][delta].append(res)
                                
                                results_by_learner_model_per_policy[policy][key_lm][delta].append(res)

        # Final stats computation
        learner_model_stats_non_random = {
            k: [(compute_avg_se(v[delta])) for delta in delta_list]
            for k, v in results_by_learner_model_non_random.items()
        }
        learner_stats_non_random = {
            k: [(compute_avg_se(v[delta])) for delta in delta_list]
            for k, v in results_by_learner_non_random.items()
        }
        model_stats_non_random = {
            k: [(compute_avg_se(v[delta])) for delta in delta_list]
            for k, v in results_by_model_non_random.items()
        }

        learner_model_stats_random = {
            k: [(compute_avg_se(v[delta])) for delta in delta_list]
            for k, v in results_by_learner_model_random.items()
        }
        learner_stats_random = {
            k: [(compute_avg_se(v[delta])) for delta in delta_list]
            for k, v in results_by_learner_random.items()
        }
        model_stats_random = {
            k: [(compute_avg_se(v[delta])) for delta in delta_list]
            for k, v in results_by_model_random.items()
        }

        learner_model_stats_per_policy = {
            k: {learner_model: [(compute_avg_se(v[delta])) for delta in delta_list] for learner_model, v in v.items()}
            for k, v in results_by_learner_model_per_policy.items()
        }

        # Save the results
        save_data(path=path + "\\res\\SimBank\\learner_model_stats_non_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl", data=learner_model_stats_non_random)
        save_data(path=path + "\\res\\SimBank\\learner_stats_non_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl", data=learner_stats_non_random)
        save_data(path=path + "\\res\\SimBank\\model_stats_non_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl", data=model_stats_non_random)

        save_data(path=path + "\\res\\SimBank\\learner_model_stats_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl", data=learner_model_stats_random)
        save_data(path=path + "\\res\\SimBank\\learner_stats_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl", data=learner_stats_random)
        save_data(path=path + "\\res\\SimBank\\model_stats_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl", data=model_stats_random)

        if plot_each_policy:
            save_data(path=path + "\\res\\SimBank\\learner_model_stats_per_policy_" + intervention + metric + fixed_policy_delta_to_add + ".pkl", data=learner_model_stats_per_policy)
    else:
        learner_model_stats_non_random = load_data(path + "\\res\\SimBank\\learner_model_stats_non_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl")
        learner_stats_non_random = load_data(path + "\\res\\SimBank\\learner_stats_non_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl")
        model_stats_non_random = load_data(path + "\\res\\SimBank\\model_stats_non_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl")
        print('learner model stats non random', learner_model_stats_non_random)

        learner_model_stats_random = load_data(path + "\\res\\SimBank\\learner_model_stats_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl")
        learner_stats_random = load_data(path + "\\res\\SimBank\\learner_stats_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl")
        model_stats_random = load_data(path + "\\res\\SimBank\\model_stats_random_" + intervention + metric + fixed_policy_delta_to_add + ".pkl")

        if plot_each_policy:
            learner_model_stats_per_policy = load_data(path + "\\res\\SimBank\\learner_model_stats_per_policy_" + intervention + metric + fixed_policy_delta_to_add + ".pkl")
        else:
            learner_model_stats_per_policy = {}

    # print("Learner + Model Type (Non-Random):", learner_model_stats_non_random)
    # print("Learner (Non-Random):", learner_stats_non_random)
    # print("Model Type (Non-Random):", model_stats_non_random)

    # print("Learner + Model Type (Random):", learner_model_stats_random)
    # print("Learner (Random):", learner_stats_random)
    # print("Model Type (Random):", model_stats_random)

    # print("Learner + Model Type (Per Policy):", learner_model_stats_per_policy)

    # Plot learner + model_type
    addition = ""
    if metric == "wssd":
        addition = "(Random)"
        if single:
            plot_stats(metric, ax, learner_model_stats_non_random, delta_list, "Average " + metric + " by Learner + Model Type (Non-Random)" + intervention, single=single)
           

        # Now combine (so plot learner + model_type and model_type together)
        
        else:
            if to_plot == "model_stats_non_random":
                plot_stats(metric,ax, model_stats_non_random, delta_list, "Average " + metric + " by Model Type (Non-Random)", sub_stats_dict=learner_model_stats_non_random, single=single)
            if to_plot == "learner_stats_non_random":
                plot_stats(metric, ax, learner_stats_non_random, delta_list, "Average " + metric + " by Learner (Non-Random)", sub_stats_dict=learner_model_stats_non_random, single=single)

            if plot_each_policy:
                for policy in learner_model_stats_per_policy:
                    plot_stats(metric, learner_model_stats_per_policy[policy], delta_list, "Average " + metric + " by Learner + Model Type (Non-Random) for policy " + policy, single=single)

    # leave out the majority_votes
    learner_model_stats_non_random = {k: v for k, v in learner_model_stats_non_random.items() if k.split('_')[0] not in majority_votes}
    learner_model_stats_random = {k: v for k, v in learner_model_stats_random.items() if k.split('_')[0] not in majority_votes}
    learner_stats_non_random = {k: v for k, v in learner_stats_non_random.items() if k.split('_')[0] not in majority_votes}
    learner_stats_random = {k: v for k, v in learner_stats_random.items() if k.split('_')[0] not in majority_votes}
    model_stats_non_random = {k: v for k, v in model_stats_non_random.items() if k.split('_')[0] not in majority_votes}
    model_stats_random = {k: v for k, v in model_stats_random.items() if k.split('_')[0] not in majority_votes}

    if single:
        plot_stats(metric, ax, learner_model_stats_random, delta_list, "Average " + metric + " by Learner + Model Type " + addition + intervention, single=single)
    else:
        # Now combine (so plot learner + model_type and model_type together)
        if to_plot == "model_stats_random":
            plot_stats(metric, ax, model_stats_random, delta_list, "Average " + metric + " by Model Type " + addition, sub_stats_dict=learner_model_stats_random, single=single)
        if to_plot == "learner_stats_random":
            plot_stats(metric, ax, learner_stats_random, delta_list, "Average " + metric + " by Learner " + addition, sub_stats_dict=learner_model_stats_random, single=single)


def plot_stats(metric, ax, stats_dict, delta_list, title, sub_stats_dict=None, single=False):
    print("Plotting stats for metric:", metric, "with title:", title)

    import matplotlib.pyplot as plt

    if single:
        fig, ax = plt.subplots(figsize=(12, 6))
    # plt.figure(figsize=(12, 6))
    # ax.set_title(title, fontsize=14, pad=20)
    # ax.set_xlabel("Delta", fontsize=12, labelpad=10)
    # ax.set_ylabel(metric, fontsize=12, labelpad=10)
    # ax.set_xticks(delta_list)
    # ax.tick_params(axis='x', labelsize=10)
    # ax.tick_params(axis='y', labelsize=10)
    ax.grid(alpha=0.0)

    handles_primary = []
    handles_secondary = []    
    
    for key in stats_dict:
        means = [val[0] for val in stats_dict[key]]
        ses = [val[1] for val in stats_dict[key]]

        marker = marker_map.get(key.split('_')[-1].lower(), 'o')  # default marker if not found
        color = color_map.get(key.split('_')[0].lower(), 'blue')  # default color if not found
        dash_style = dash_map.get(key.split('_')[-1].lower(), "solid")  # default dash style if not found

        line, = ax.plot(delta_list, means, label=key, linewidth=2,
                        markersize=8, marker=marker, color=color, linestyle=dash_style)
        ax.fill_between(delta_list,
                         [m - se for m, se in zip(means, ses)],
                         [m + se for m, se in zip(means, ses)],
                         alpha=0.2,
                         color=color)

        # Group handles for legend (e.g., by model type if present)
        if "MLP" in key:
            handles_primary.append(line)
        elif "LSTM" in key:
            handles_secondary.append(line)
        else:
            handles_primary.append(line)  # default


    # Plot secondary stats if provided (with lower opacity)
    if sub_stats_dict:
        for key in sub_stats_dict:
            means = [val[0] for val in sub_stats_dict[key]]
            ses = [val[1] for val in sub_stats_dict[key]]

            marker = marker_map.get(key.split('_')[-1].lower(), 'o')
            color = color_map.get(key.split('_')[0].lower(), 'blue')  # default color if not found
            dash_style = dash_map.get(key.split('_')[-1].lower(), "solid")  # default dash style if not found

            opacity = [0.4, 0]
            if dash_style == "solid":
                opacity = [0.2, 0]
            
            if metric =="wssd":
                opacity = [0.5, 0]
                if dash_style == "solid":
                    opacity = [0.3, 0]

            line, = ax.plot(delta_list, means, label=key, linewidth=2,
                            markersize=8, marker=marker, color=color, linestyle=dash_style, alpha=opacity[0])
            ax.fill_between(delta_list,
                            [m - se for m, se in zip(means, ses)],
                            [m + se for m, se in zip(means, ses)],
                            alpha=opacity[1],
                            color=color)
            
            if "MLP" in key:
                handles_primary.append(line)
            elif "LSTM" in key:
                handles_secondary.append(line)
            else:
                handles_primary.append(line)

    if single:
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel("Delta", fontsize=12, labelpad=10)
        ax.set_ylabel(metric, fontsize=12, labelpad=10)
        
        # make legend
        if len(handles_primary) > 0:
            ax.legend(handles=handles_primary, 
                      labels=[h.get_label() for h in handles_primary],
                      fontsize=10, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.15), frameon=False, ncol=3)
        if len(handles_secondary) > 0:
            ax.gca().add_artist(ax.legend(handles=handles_secondary, 
                                          labels=[h.get_label() for h in handles_secondary],
                                          fontsize=10, loc='upper center', 
                                          bbox_to_anchor=(0.5, -0.25), frameon=False, ncol=3))

        # show the plot
        plt.show()

        # save the figure
        fig.savefig(path + "\\res\\SimBank\\non_aggregated_" + title.replace(" ", "_") + ".png", bbox_inches='tight', dpi=300)

    # # Legend formatting
    # # Organize legends in two horizontal layers
    # # First layer: primary handles
    # # Second layer: secondary handles
    # if len(handles_primary) > 0:
    #     plt.legend(handles=handles_primary, 
    #               labels=[h.get_label() for h in handles_primary],
    #               fontsize=10, loc='upper center', 
    #               bbox_to_anchor=(0.0, -0.15), frameon=False, ncol=3)
    
    # # Add second legend for the secondary handles if they exist
    # if len(handles_secondary) > 0:
    #     plt.gca().add_artist(plt.legend(handles=handles_secondary, 
    #                                   labels=[h.get_label() for h in handles_secondary],
    #                                   fontsize=10, loc='upper center', 
    #                                   bbox_to_anchor=(0.0, -0.25), frameon=False, ncol=3))
    
    # # Adjust the bottom margin to make room for both legend layers
    # plt.tight_layout(rect=[0, 0.15, 1, 1])  # Increase bottom margin for two legend rows
    # plt.show()

    # legend_handles = handles_primary + handles_secondary
    # legend_labels = [h.get_label() for h in legend_handles]
    # plt.legend(legend_handles, legend_labels, fontsize=10, loc='upper left',
    #            frameon=False, ncol=2)
    # plt.tight_layout()
    # plt.show()



def get_pred_results(learners, intervention, num_iterations_generator, num_iterations_eval, policies=["all"], delta=0.95, calculate=True):
    from sklearn.metrics import mean_absolute_error as mae

    pred_metrics_pc = {}
    pred_metrics_rc = {}

    # check whether the file already exists, otherwise no need to calculate it again
    if os.path.exists(path + "\\res\\SimBank\\pred_metrics_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl") and not calculate:
       pred_metrics_pc = load_data(path + "\\res\\SimBank\\pred_metrics_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
       pred_metrics_rc = load_data(path + "\\res\\SimBank\\pred_metrics_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
       return pred_metrics_pc, pred_metrics_rc
    
    # true_outcome_dfs_path = "\\res\\SimBank\\offine_outcome_dfs_" + intervention + str(delta) + "_generator_iteration" + str(0) + "RealCause" + "['all']"
    # # check if the file exists, if not then it should be with 'all' in the name
    # if not os.path.exists(path + true_outcome_dfs_path):
    #     true_outcome_dfs_path = "\\res\\SimBank\\offine_outcome_dfs_" + intervention + str(delta) + "_generator_iteration" + str(0) + "RealCause" + str(policies)
    # true_outcome_dfs = load_data(path + true_outcome_dfs_path)
    # bank_test_set = true_outcome_dfs["bank"]

    bank_test_set = load_data(path + "\\res\\SimBank\\bank_test_df_" + intervention +"0.95" + "_generator_iteration" + str(0) + "RealCause")

    for learner in learners + ["ensemble"]:
        pred_metrics_pc[learner] = 0
        pred_metrics_rc[learner] = 0
        for iter_gen in range(num_iterations_generator):
            path_pc = "\\res\\SimBank\\estimated_outcome_dfs_procause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + "['all']" + fixed_policy_delta_to_add
            path_rc = "\\res\\SimBank\\estimated_outcome_dfs_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + "['all']" + fixed_policy_delta_to_add
            if not os.path.exists(path + path_pc):
                path_pc = "\\res\\SimBank\\estimated_outcome_dfs_procause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "ProCause" + learner + str(policies) + fixed_policy_delta_to_add
            if not os.path.exists(path + path_rc):
                path_rc = "\\res\\SimBank\\estimated_outcome_dfs_realcause_" + intervention + str(delta) + "_generator_iteration" + str(iter_gen) + "RealCause" + learner + str(policies) + fixed_policy_delta_to_add
            
            # for iter_eval in range(num_iterations_eval):
            df_rc = load_data(path + path_rc)
            df_rc = df_rc["bank"][0]
            df_pc = load_data(path + path_pc)
            df_pc = df_pc["bank"][0]
            mse_pc = 0
            mse_rc = 0
            for case_nr in bank_test_set["case_nr"].unique():
                if case_nr not in df_rc["case_nr"].unique():
                    continue
                # get the last outcome for the case_nr in the bank_test_set
                true_outcome = bank_test_set[bank_test_set["case_nr"] == case_nr]["outcome"].iloc[-1]
                # there is 1 outcome for the bank_test_set, while 50 corresponding ones for the df_pc and df_rc, so calculate the mse for each of them and then take the mean
                estimated_outcomes_pc = df_pc[df_pc["case_nr"] == case_nr]
                estimated_outcomes_rc = df_rc[df_rc["case_nr"] == case_nr]
                # repeat the true_outcome for the length of the estimated_outcomes_pc and estimated_outcomes_rc
                true_outcome = pd.DataFrame({"case_nr": estimated_outcomes_pc["case_nr"].values, "outcome": [true_outcome] * len(estimated_outcomes_pc)})
                mse_pc += mae(true_outcome["outcome"], estimated_outcomes_pc["outcome"])
                mse_rc += mae(true_outcome["outcome"], estimated_outcomes_rc["outcome"])
            
            pred_metrics_pc[learner] += mse_pc / len(bank_test_set["case_nr"].unique())
            pred_metrics_rc[learner] += mse_rc / len(bank_test_set["case_nr"].unique())
        
        pred_metrics_pc[learner] /= num_iterations_generator
        pred_metrics_rc[learner] /= num_iterations_generator

    save_data(path=path + "\\res\\SimBank\\pred_metrics_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=pred_metrics_pc)
    save_data(path=path + "\\res\\SimBank\\pred_metrics_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl", data=pred_metrics_rc)

    print("Pred metrics ProCause: ", pred_metrics_pc)
    print("Pred metrics RealCause: ", pred_metrics_rc)

    return pred_metrics_pc, pred_metrics_rc

def plot_delta_wssd_results(full_dict_wssd_list, full_dict_conf_int_list, intervention, delta_list, policies=["all"]):
    if "all" in policies:
        policies = ["random", "S-Learner_LSTM", "T-Learner_LSTM", "TarNet_LSTM", "S-Learner_Vanilla_NN", "T-Learner_Vanilla_NN", "TarNet_Vanilla_NN"]
    
    print(full_dict_wssd_list)
    for i in range(len(full_dict_wssd_list)):
        if "Ensemble ProCause" not in full_dict_wssd_list[i]:
            continue
        print(full_dict_conf_int_list[i])
        full_dict_wssd_list[i]["Ensemble-LSTM"] = full_dict_wssd_list[i]["Ensemble ProCause"]
        full_dict_wssd_list[i]["Ensemble-MLP"] = full_dict_wssd_list[i]["Ensemble RealCause"]
        full_dict_conf_int_list[i]["Ensemble-LSTM"] = full_dict_conf_int_list[i]["Ensemble ProCause"]
        full_dict_conf_int_list[i]["Ensemble-MLP"] = full_dict_conf_int_list[i]["Ensemble RealCause"]
        # delete the old keys
        del full_dict_wssd_list[i]["Ensemble ProCause"]
        del full_dict_wssd_list[i]["Ensemble RealCause"]
        del full_dict_conf_int_list[i]["Ensemble ProCause"]
        del full_dict_conf_int_list[i]["Ensemble RealCause"]


    # get standard error by dividing by 1.96
    for i in range(len(full_dict_conf_int_list)):
        for key in full_dict_conf_int_list[i]:
            for policy in full_dict_conf_int_list[i][key]:
                full_dict_conf_int_list[i][key][policy] = full_dict_conf_int_list[i][key][policy] / 1.96

    for policy in policies:
        if policy == "bank":
            continue
        values_lists = {
            'TarNet MLP': [full_dict_wssd_list[i]["TarNet-MLP"][policy] for i in range(len(full_dict_wssd_list))],
            'Ensemble LSTM': [full_dict_wssd_list[i]["Ensemble-LSTM"][policy] for i in range(len(full_dict_wssd_list))],
            'Ensemble MLP': [full_dict_wssd_list[i]["Ensemble-MLP"][policy] for i in range(len(full_dict_wssd_list))]
        }

        conf_int_lists = {
            'TarNet MLP': [full_dict_conf_int_list[i]["TarNet-MLP"][policy] for i in range(len(full_dict_conf_int_list))],
            'Ensemble LSTM': [full_dict_conf_int_list[i]["Ensemble-LSTM"][policy] for i in range(len(full_dict_conf_int_list))],
            'Ensemble MLP': [full_dict_conf_int_list[i]["Ensemble-MLP"][policy] for i in range(len(full_dict_conf_int_list))]
        }

        print("Policy:", policy)
        print("Values:", values_lists)
        print("Confidence intervals:", conf_int_lists)

        # Plot a line plot for each policy separately, with enough spacing between elements
        # plt.figure(figsize=(12, 6))
        # plt.title(f"WSSD for {policy} with different delta values", fontsize=14, pad=20)
        # plt.xlabel("Delta", fontsize=12, labelpad=10)
        # plt.ylabel("WSSD", fontsize=12, labelpad=10)
        # plt.xticks(delta_list, fontsize=10)
        # plt.yticks(fontsize=10)
        # plt.grid(alpha=0.3)

        # # Plot each learner and ensemble with confidence intervals
        # for key, values in values_lists.items():
        #     index = list(values_lists.keys()).index(key)
        #     color = plt.cm.viridis(index / len(values_lists))
        #     plt.plot(delta_list, values, label=key, color=color, linewidth=2, marker='o', markersize=8)
        #     plt.fill_between(delta_list, 
        #                      [values[i] - conf_int_lists[key][i] for i in range(len(values))], 
        #                      [values[i] + conf_int_lists[key][i] for i in range(len(values))], 
        #                      alpha=0.2, color=color)
        
        # plt.legend(fontsize=10, loc='upper left', frameon=False)
        # plt.tight_layout()
        # plt.show()

    # Calculate average WSSD values across all policies (excluding "random")
    average_wssd = {}
    average_conf_int = {}

    for key in full_dict_wssd_list[0]:
        average_wssd[key] = [
            np.mean([full_dict_wssd_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"] if policy != "random"])
            for i in range(len(full_dict_wssd_list))
        ]
        average_conf_int[key] = [
            np.mean([full_dict_conf_int_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"] if policy != "random"])
            for i in range(len(full_dict_conf_int_list))
        ]

    # Plot average WSSD across policies
    plt.figure(figsize=(12, 6))
    plt.title("Average WSSD Across Policies (Excluding Random)", fontsize=14, pad=20)
    plt.xlabel("Delta", fontsize=12, labelpad=10)
    plt.ylabel("WSSD", fontsize=12, labelpad=10)
    plt.xticks(delta_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.3)

    # Define markers based on the key name
    marker_dict = {
        "MLP": "x",  # Square for MLP
        "LSTM": "o",  # Circle for LSTM
    }

    handles_mlp = []
    handles_lstm = []

    for key in average_wssd:
        # Determine the marker based on the key name
        marker = None
        for marker_key, marker_value in marker_dict.items():
            if marker_key in key:
                marker = marker_value
                break

        line, = plt.plot(delta_list, average_wssd[key], label=key, linewidth=2, marker=marker, markersize=8)
        plt.fill_between(delta_list, 
                         [average_wssd[key][i] - average_conf_int[key][i] for i in range(len(delta_list))],
                         [average_wssd[key][i] + average_conf_int[key][i] for i in range(len(delta_list))],
                         alpha=0.2)
        
        # Add to the appropriate legend list
        if "MLP" in key:
            handles_mlp.append(line)
        elif "LSTM" in key:
            handles_lstm.append(line)
    
    # Create a legend with two columns: one for MLP and one for LSTM
    plt.legend(handles=handles_mlp + handles_lstm, 
               labels=[handle.get_label() for handle in handles_mlp] + [handle.get_label() for handle in handles_lstm],
               fontsize=10, loc='upper left', frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()

    # Calculate average WSSD values across all policies (excluding "random")
    average_wssd = {}
    average_conf_int = {}

    for key in full_dict_wssd_list[0]:
        average_wssd[key] = [
            np.mean([full_dict_wssd_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"] if policy == "random"])
            for i in range(len(full_dict_wssd_list))
        ]
        average_conf_int[key] = [
            np.mean([full_dict_conf_int_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"] if policy == "random"])
            for i in range(len(full_dict_conf_int_list))
        ]

    # Plot average WSSD across policies
    plt.figure(figsize=(12, 6))
    plt.title("Average WSSD Across the Random Policy", fontsize=14, pad=20)
    plt.xlabel("Delta", fontsize=12, labelpad=10)
    plt.ylabel("WSSD", fontsize=12, labelpad=10)
    plt.xticks(delta_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.3)

    handles_mlp = []
    handles_lstm = []

    for key in average_wssd:
        marker = None
        for marker_key, marker_value in marker_dict.items():
            if marker_key in key:
                marker = marker_value
                break
        line, = plt.plot(delta_list, average_wssd[key], label=key, linewidth=2, markersize=8, marker=marker)
        plt.fill_between(delta_list, 
                         [average_wssd[key][i] - average_conf_int[key][i] for i in range(len(delta_list))],
                         [average_wssd[key][i] + average_conf_int[key][i] for i in range(len(delta_list))],
                         alpha=0.2)
        
        # Add to the appropriate legend list
        if "MLP" in key:
            handles_mlp.append(line)
        elif "LSTM" in key:
            handles_lstm.append(line)
    
    # Create a legend with two columns: one for MLP and one for LSTM
    plt.legend(handles=handles_mlp + handles_lstm, 
               labels=[handle.get_label() for handle in handles_mlp] + [handle.get_label() for handle in handles_lstm],
               fontsize=10, loc='upper left', frameon=False, ncol=2)
    plt.tight_layout()
    plt.show()

 
    # now plot the results aggregated per learner (so MLP and LSTM aggregated for each learner)
    values_lists = {
        "TarNet": np.mean([average_wssd["TarNet-MLP"], average_wssd["TarNet-LSTM"]], axis=0),
        "S-Learner": np.mean([average_wssd["S-Learner-MLP"], average_wssd["S-Learner-LSTM"]], axis=0),
        "T-Learner": np.mean([average_wssd["T-Learner-MLP"], average_wssd["T-Learner-LSTM"]], axis=0),
        "Ensemble": np.mean([average_wssd["Ensemble-MLP"], average_wssd["Ensemble-LSTM"]], axis=0)
    }

    # calculate the new standard errors (which is 0.5 * square root of the sum of the squares of the standard errors of the two learners)
    conf_int_lists = {
        "TarNet": 0.5 * np.sqrt(np.square(average_conf_int["TarNet-MLP"]) + np.square(average_conf_int["TarNet-LSTM"])),
        "S-Learner": 0.5 * np.sqrt(np.square(average_conf_int["S-Learner-MLP"]) + np.square(average_conf_int["S-Learner-LSTM"])),
        "T-Learner": 0.5 * np.sqrt(np.square(average_conf_int["T-Learner-MLP"]) + np.square(average_conf_int["T-Learner-LSTM"])),
        "Ensemble": 0.5 * np.sqrt(np.square(average_conf_int["Ensemble-MLP"]) + np.square(average_conf_int["Ensemble-LSTM"]))
    }

    # plot these values
    plt.figure(figsize=(12, 6))
    plt.title("Average WSSD Across Learners (Excluding Random)", fontsize=14, pad=20)
    plt.xlabel("Delta", fontsize=12, labelpad=10)
    plt.ylabel("WSSD", fontsize=12, labelpad=10)
    plt.xticks(delta_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.3)

    for key in values_lists:
        marker = None
        for marker_key, marker_value in marker_dict.items():
            if marker_key in key:
                marker = marker_value
                break
        line, = plt.plot(delta_list, values_lists[key], label=key, linewidth=2, markersize=8, marker=marker)
        plt.fill_between(delta_list, 
                         [values_lists[key][i] - conf_int_lists[key][i] for i in range(len(delta_list))],
                         [values_lists[key][i] + conf_int_lists[key][i] for i in range(len(delta_list))],
                         alpha=0.2)
    
    plt.legend(fontsize=10, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()

    
def plot_delta_pred_results(pred_metrics_pc_list, pred_metrics_rc_list, delta_list, full_dict_wssd_list, full_dict_conf_int_list, spearmann_df_list, kendall_df_list):
    print(pred_metrics_pc_list)
    print(pred_metrics_rc_list)
    print(len(pred_metrics_pc_list))
    print(pred_metrics_pc_list[0]["TarNet"])

    average_wssd = {}
    average_conf_int = {}

    print("Full dict wssd list:", full_dict_wssd_list)

    # for key in full_dict_wssd_list[0]:
    #     average_wssd[key] = [
    #         np.mean([full_dict_wssd_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"]])
    #         for i in range(len(full_dict_wssd_list))
    #     ]
    #     average_conf_int[key] = [
    #         np.mean([full_dict_conf_int_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"]])
    #         for i in range(len(full_dict_conf_int_list))
    #     ]

        # average_wssd[key] = [
        #     np.mean([full_dict_wssd_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"] if policy == "random"])
        #     for i in range(len(full_dict_wssd_list))
        # ]
        # average_conf_int[key] = [
        #     np.mean([full_dict_conf_int_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"] if policy == "random"])
        #     for i in range(len(full_dict_conf_int_list))
        # ]

    for i in range(len(pred_metrics_pc_list)):
        if "ensemble" not in pred_metrics_pc_list[i]:
            continue
        pred_metrics_pc_list[i]["Ensemble"] = pred_metrics_pc_list[i]["ensemble"]
        pred_metrics_rc_list[i]["Ensemble"] = pred_metrics_rc_list[i]["ensemble"]
        # delete the old keys
        del pred_metrics_pc_list[i]["ensemble"]
        del pred_metrics_rc_list[i]["ensemble"]

    values_lists = {
        'TarNet-LSTM': [pred_metrics_pc_list[i]["TarNet"] for i in range(len(pred_metrics_pc_list))],
        'S-Learner-LSTM': [pred_metrics_pc_list[i]["S-Learner"] for i in range(len(pred_metrics_pc_list))],
        'T-Learner-LSTM': [pred_metrics_pc_list[i]["T-Learner"] for i in range(len(pred_metrics_pc_list))],
        'TarNet-MLP': [pred_metrics_rc_list[i]["TarNet"] for i in range(len(pred_metrics_rc_list))],
        'S-Learner-MLP': [pred_metrics_rc_list[i]["S-Learner"] for i in range(len(pred_metrics_rc_list))],
        'T-Learner-MLP': [pred_metrics_rc_list[i]["T-Learner"] for i in range(len(pred_metrics_rc_list))],
        'Ensemble-LSTM': [pred_metrics_pc_list[i]["Ensemble"] for i in range(len(pred_metrics_pc_list))],
        'Ensemble-MLP': [pred_metrics_rc_list[i]["Ensemble"] for i in range(len(pred_metrics_rc_list))]
    }

    print("Values:", values_lists)

    # Define markers based on the key name
    marker_dict = {
        "MLP": "x",  # Square for MLP
        "LSTM": "o",  # Circle for LSTM
    }

    # Plot a line plot with enough spacing between elements
    plt.figure(figsize=(12, 6))
    plt.title("MAE on factuals with different delta values", fontsize=14, pad=20)
    plt.xlabel("Delta", fontsize=12, labelpad=10)
    plt.ylabel("MAE", fontsize=12, labelpad=10)
    plt.xticks(delta_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.3)

    # Plot
    for key, values in values_lists.items():
        marker = None
        for marker_key, marker_value in marker_dict.items():
            if marker_key in key:
                marker = marker_value
                break
        plt.plot(delta_list, values, label=key, linewidth=2, marker=marker, markersize=8)
    
    plt.legend(fontsize=10, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()


    # Add a plot comparing LSTM predictive performance, LSTM WSSD performance, MLP predictive performance, and MLP WSSD performance
    # quickly make a new average_wssd without the "random" policy
    # avg_wssd_non_random = {}
    # for key in average_wssd:
    #     avg_wssd_non_random[key] = [
    #         np.mean([full_dict_wssd_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"] if policy != "random"])
    #         for i in range(len(full_dict_wssd_list))
    #     ]
    lstm_pred_values = [np.mean([pred_metrics_pc_list[i][method] for method in pred_metrics_pc_list[i]]) for i in range(len(delta_list))]
    mlp_pred_values = [np.mean([pred_metrics_rc_list[i][method] for method in pred_metrics_rc_list[i]]) for i in range(len(delta_list))]
    # lstm_wssd_values = [np.mean([avg_wssd_non_random[method + "-LSTM"][i] for method in pred_metrics_pc_list[i]]) for i in range(len(delta_list))]
    # mlp_wssd_values = [np.mean([avg_wssd_non_random[method + "-MLP"][i] for method in pred_metrics_rc_list[i]]) for i in range(len(delta_list))]

    # Extract confidence intervals for WSSD
    # lstm_wssd_ci = [np.mean([full_dict_conf_int_list[i][method + "-LSTM"]["random"] for method in pred_metrics_pc_list[i]]) for i in range(len(delta_list))]
    # mlp_wssd_ci = [np.mean([full_dict_conf_int_list[i][method + "-MLP"]["random"] for method in pred_metrics_rc_list[i]]) for i in range(len(delta_list))]

    plt.figure(figsize=(12, 6))
    plt.title("Comparison of Predictive and Causal Performance", fontsize=14, pad=20)
    plt.xlabel("Delta", fontsize=12, labelpad=10)
    plt.ylabel("Performance", fontsize=12, labelpad=10)
    plt.xticks(delta_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.3)

    # Plot with confidence intervals
    plt.plot(delta_list, lstm_pred_values, label="LSTM Predictive Performance", linewidth=2, marker="o", markersize=8, color="blue")

    # plt.plot(delta_list, lstm_wssd_values, label="LSTM Causal Performance", linewidth=2, marker="x", markersize=8, color="cyan")
    # plt.fill_between(delta_list, 
    #                  [lstm_wssd_values[i] - lstm_wssd_ci[i] for i in range(len(delta_list))],
    #                  [lstm_wssd_values[i] + lstm_wssd_ci[i] for i in range(len(delta_list))],
    #                  color="cyan", alpha=0.2)

    plt.plot(delta_list, mlp_pred_values, label="MLP Predictive Performance", linewidth=2, marker="o", markersize=8, color="red")

    # plt.plot(delta_list, mlp_wssd_values, label="MLP Causal Performance", linewidth=2, marker="x", markersize=8, color="orange")
    # plt.fill_between(delta_list, 
    #                  [mlp_wssd_values[i] - mlp_wssd_ci[i] for i in range(len(delta_list))],
    #                  [mlp_wssd_values[i] + mlp_wssd_ci[i] for i in range(len(delta_list))],
    #                  color="orange", alpha=0.2)

    plt.legend(fontsize=10, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()

    # Calculate whether there is any correlation between the WSSD and the prediction metrics (one correlation for each delta value over all methods)
    wssd_pred_corr = {}
    wssd_pred_corr_spearman = {}
    for i, delta in enumerate(delta_list):
        wssd_values = []
        pred_values = []
        for method in pred_metrics_pc_list[i]:
            method_pc = method + "-LSTM"
            method_rc = method + "-MLP"
            if method_pc in full_dict_wssd_list[i] and method_rc in full_dict_wssd_list[i]:
                # wssd_values.append((average_wssd[method_pc][i] + average_wssd[method_rc][i]) / 2)
                # pred_values.append((pred_metrics_pc_list[i][method] + pred_metrics_rc_list[i][method]) / 2)
                wssd_values.append(average_wssd[method_pc][i])
                wssd_values.append(average_wssd[method_rc][i])
                pred_values.append(pred_metrics_pc_list[i][method])
                pred_values.append(pred_metrics_rc_list[i][method])
        # Calculate correlation
        if len(wssd_values) > 1 and len(pred_values) > 1:
            corr, _ = scipy.stats.pearsonr(wssd_values, pred_values)
            wssd_pred_corr[delta] = corr
            corr_spearman, _ = scipy.stats.spearmanr(wssd_values, pred_values)
            wssd_pred_corr_spearman[delta] = corr_spearman
        else:
            wssd_pred_corr[delta] = None
            wssd_pred_corr_spearman[delta] = None


    # delete all spaces from the columns in the spearmann_df_list and kendall_df_list
    for i in range(len(spearmann_df_list)):
        spearmann_df_list[i].columns = spearmann_df_list[i].columns.str.replace(" ", "")
        kendall_df_list[i].columns = kendall_df_list[i].columns.str.replace(" ", "")

    # Normalize predictive and causal performance values for comparability
    lstm_pred_values = [np.mean([pred_metrics_pc_list[i][method] for method in pred_metrics_pc_list[i]]) for i in range(len(delta_list))]
    mlp_pred_values = [np.mean([pred_metrics_rc_list[i][method] for method in pred_metrics_rc_list[i]]) for i in range(len(delta_list))]
    lstm_spearman_values = [np.mean([spearmann_df_list[i][method + "-LSTM"].iloc[0] for method in pred_metrics_pc_list[i]]) for i in range(len(delta_list))]
    mlp_spearman_values = [np.mean([spearmann_df_list[i][method + "-MLP"].iloc[0] for method in pred_metrics_rc_list[i]]) for i in range(len(delta_list))]

    # Normalize values to [0, 1] range
    lstm_pred_values = (np.array(lstm_pred_values) - np.min(lstm_pred_values)) / (np.max(lstm_pred_values) - np.min(lstm_pred_values))
    mlp_pred_values = (np.array(mlp_pred_values) - np.min(mlp_pred_values)) / (np.max(mlp_pred_values) - np.min(mlp_pred_values))
    lstm_spearman_values = (np.array(lstm_spearman_values) - np.min(lstm_spearman_values)) / (np.max(lstm_spearman_values) - np.min(lstm_spearman_values))
    mlp_spearman_values = (np.array(mlp_spearman_values) - np.min(mlp_spearman_values)) / (np.max(mlp_spearman_values) - np.min(mlp_spearman_values))

    plt.figure(figsize=(12, 6))
    plt.title("Comparison of Predictive and Causal Performance (Spearman)", fontsize=14, pad=20)
    plt.xlabel("Delta", fontsize=12, labelpad=10)
    plt.ylabel("Normalized Performance", fontsize=12, labelpad=10)
    plt.xticks(delta_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.3)

    plt.plot(delta_list, lstm_pred_values, label="LSTM Predictive Performance", linewidth=2, marker="o", markersize=8, color="blue")
    plt.plot(delta_list, lstm_spearman_values, label="LSTM Causal Performance (Spearman)", linewidth=2, marker="x", markersize=8, color="cyan")
    plt.plot(delta_list, mlp_pred_values, label="MLP Predictive Performance", linewidth=2, marker="o", markersize=8, color="red")
    plt.plot(delta_list, mlp_spearman_values, label="MLP Causal Performance (Spearman)", linewidth=2, marker="x", markersize=8, color="orange")

    plt.legend(fontsize=10, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()

    # Add a plot comparing LSTM predictive performance vs Kendall causal performance
    lstm_kendall_values = [np.mean([kendall_df_list[i][method + "-LSTM"].iloc[0] for method in pred_metrics_pc_list[i]]) for i in range(len(delta_list))]
    mlp_kendall_values = [np.mean([kendall_df_list[i][method + "-MLP"].iloc[0] for method in pred_metrics_rc_list[i]]) for i in range(len(delta_list))]

    # Normalize Kendall values
    lstm_kendall_values = (np.array(lstm_kendall_values) - np.min(lstm_kendall_values)) / (np.max(lstm_kendall_values) - np.min(lstm_kendall_values))
    mlp_kendall_values = (np.array(mlp_kendall_values) - np.min(mlp_kendall_values)) / (np.max(mlp_kendall_values) - np.min(mlp_kendall_values))

    plt.figure(figsize=(12, 6))
    plt.title("Comparison of Predictive and Causal Performance (Kendall)", fontsize=14, pad=20)
    plt.xlabel("Delta", fontsize=12, labelpad=10)
    plt.ylabel("Normalized Performance", fontsize=12, labelpad=10)
    plt.xticks(delta_list, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(alpha=0.3)

    plt.plot(delta_list, lstm_pred_values, label="LSTM Predictive Performance", linewidth=2, marker="o", markersize=8, color="blue")
    plt.plot(delta_list, lstm_kendall_values, label="LSTM Causal Performance (Kendall)", linewidth=2, marker="x", markersize=8, color="cyan")
    plt.plot(delta_list, mlp_pred_values, label="MLP Predictive Performance", linewidth=2, marker="o", markersize=8, color="red")
    plt.plot(delta_list, mlp_kendall_values, label="MLP Causal Performance (Kendall)", linewidth=2, marker="x", markersize=8, color="orange")

    plt.legend(fontsize=10, loc='upper left', frameon=False)
    plt.tight_layout()
    plt.show()

    print("Correlation between WSSD and prediction metrics for each delta value:")
    for delta, corr in wssd_pred_corr.items():
        print(f"Delta {delta}: Correlation = {corr}")
        print(f"Delta {delta}: Spearman Correlation = {wssd_pred_corr_spearman[delta]}")

    # Calculate the correlation between the prediction metrics and the WSSD for each method separately over all delta values
    method_corr = {}
    method_corr_spearman = {}
    for method in pred_metrics_pc_list[0]:
        wssd_values = []
        pred_values = []
        for i in range(len(delta_list)):
            method_pc = method + "-LSTM"
            method_rc = method + "-MLP"
            if method_pc in full_dict_wssd_list[i] and method_rc in full_dict_wssd_list[i]:
                # wssd_values.append((average_wssd[method_pc][i] + average_wssd[method_rc][i]) / 2)
                # pred_values.append((pred_metrics_pc_list[i][method] + pred_metrics_rc_list[i][method]) / 2)
                wssd_values.append(average_wssd[method_pc][i])
                wssd_values.append(average_wssd[method_rc][i])
                pred_values.append(pred_metrics_pc_list[i][method])
                pred_values.append(pred_metrics_rc_list[i][method])
        # Calculate correlation
        if len(wssd_values) > 1 and len(pred_values) > 1:
            corr, _ = scipy.stats.pearsonr(wssd_values, pred_values)
            method_corr[method] = corr
            corr_spearman, _ = scipy.stats.spearmanr(wssd_values, pred_values)
            method_corr_spearman[method] = corr_spearman
        else:
            method_corr[method] = None
            method_corr_spearman[method] = None

    print("\nCorrelation between prediction metrics and WSSD for each method across all delta values:")
    for method, corr in method_corr.items():
        print(f"Method {method}: Correlation = {corr}")
        print(f"Method {method}: Spearman Correlation = {method_corr_spearman[method]}")

    # now calculate one correlation over all methods and all delta values
    wssd_values = []
    pred_values = []
    for i in range(len(delta_list)):
        for method in pred_metrics_pc_list[i]:
            method_pc = method + "-LSTM"
            method_rc = method + "-MLP"
            if method_pc in full_dict_wssd_list[i] and method_rc in full_dict_wssd_list[i]:
                # wssd_values.append((average_wssd[method_pc][i] + average_wssd[method_rc][i]) / 2)
                # pred_values.append((pred_metrics_pc_list[i][method] + pred_metrics_rc_list[i][method]) / 2)
                wssd_values.append(average_wssd[method_pc][i])
                wssd_values.append(average_wssd[method_rc][i])
                pred_values.append(pred_metrics_pc_list[i][method])
                pred_values.append(pred_metrics_rc_list[i][method])
    # Calculate correlation
    if len(wssd_values) > 1 and len(pred_values) > 1:
        corr, _ = scipy.stats.pearsonr(wssd_values, pred_values)
        wssd_pred_corr["all"] = corr
        corr_spearman, _ = scipy.stats.spearmanr(wssd_values, pred_values)
        wssd_pred_corr_spearman["all"] = corr_spearman
    else:
        wssd_pred_corr["all"] = None
        wssd_pred_corr_spearman["all"] = None
    
    print("\nCorrelation between WSSD and prediction metrics for all methods and delta values:")
    print(f"All methods: Correlation = {wssd_pred_corr['all']}")
    print(f"All methods: Spearman Correlation = {wssd_pred_corr_spearman['all']}")

    # do the same as the previous but for random policy only (so get the average WSSD for random policy and the average prediction metrics for random policy)
    wssd_values = []
    pred_values = []
    avg_random_wssd = {}
    for key in full_dict_wssd_list[0]:
        avg_random_wssd[key] = [
            full_dict_wssd_list[i][key]["random"] for i in range(len(full_dict_wssd_list))
        ]
    for i in range(len(delta_list)):
        for method in pred_metrics_pc_list[i]:
            method_pc = method + "-LSTM"
            method_rc = method + "-MLP"
            if method_pc in full_dict_wssd_list[i] and method_rc in full_dict_wssd_list[i]:
    #             wssd_values.append(avg_random_wssd[method_pc][i] + avg_random_wssd[method_rc][i] / 2)
    #             pred_values.append((pred_metrics_pc_list[i][method] + pred_metrics_rc_list[i][method]) / 2)
                wssd_values.append(avg_random_wssd[method_pc][i])
                wssd_values.append(avg_random_wssd[method_rc][i])
                pred_values.append(pred_metrics_pc_list[i][method])
                pred_values.append(pred_metrics_rc_list[i][method])
    # # Calculate correlation
    if len(wssd_values) > 1 and len(pred_values) > 1:
        corr, _ = scipy.stats.pearsonr(wssd_values, pred_values)
        wssd_pred_corr["random"] = corr
        corr_spearman, _ = scipy.stats.spearmanr(wssd_values, pred_values)
        wssd_pred_corr_spearman["random"] = corr_spearman
    else:
        wssd_pred_corr["random"] = None
        wssd_pred_corr_spearman["random"] = None
    
    print("\nCorrelation between WSSD and prediction metrics for random policy:")
    print(f"Random policy: Correlation = {wssd_pred_corr['random']}")
    print(f"Random policy: Spearman Correlation = {wssd_pred_corr_spearman['random']}")

    # do the same as the previous but for all policies except the random one
    wssd_values = []
    pred_values = []
    avg_non_random_wssd = {}
    for key in full_dict_wssd_list[0]:
        avg_non_random_wssd[key] = [
            np.mean([full_dict_wssd_list[i][key][policy] for policy in full_dict_wssd_list[0]["TarNet-LSTM"] if policy != "random"])
            for i in range(len(full_dict_wssd_list))
        ]
    for i in range(len(delta_list)):
        for method in pred_metrics_pc_list[i]:
            method_pc = method + "-LSTM"
            method_rc = method + "-MLP"
            if method_pc in full_dict_wssd_list[i] and method_rc in full_dict_wssd_list[i]:
                # wssd_values.append(avg_non_random_wssd[method_pc][i] + avg_non_random_wssd[method_rc][i] / 2)
                # pred_values.append((pred_metrics_pc_list[i][method] + pred_metrics_rc_list[i][method]) / 2)
                wssd_values.append(avg_non_random_wssd[method_pc][i])
                wssd_values.append(avg_non_random_wssd[method_rc][i])
                pred_values.append(pred_metrics_pc_list[i][method])
                pred_values.append(pred_metrics_rc_list[i][method])
    # Calculate correlation
    if len(wssd_values) > 1 and len(pred_values) > 1:
        corr, _ = scipy.stats.pearsonr(wssd_values, pred_values)
        wssd_pred_corr["non_random"] = corr
        corr_spearman, _ = scipy.stats.spearmanr(wssd_values, pred_values)
        wssd_pred_corr_spearman["non_random"] = corr_spearman
    else:
        wssd_pred_corr["non_random"] = None
        wssd_pred_corr_spearman["non_random"] = None
    
    print("\nCorrelation between WSSD and prediction metrics for all policies except random:")
    print(f"Non-random policies: Correlation = {wssd_pred_corr['non_random']}")
    print(f"Non-random policies: Spearman Correlation = {wssd_pred_corr_spearman['non_random']}")

    # Now do all of the above for the spearmann and kendall tau correlation matrices
    wssd_values = []
    pred_values = []
    for i in range(len(delta_list)):
        for method in pred_metrics_pc_list[i]:
            method_pc = " " + method + "-LSTM"
            method_rc = " " + method + "-MLP"
            if method_pc in spearmann_df_list[i] and method_rc in spearmann_df_list[i]:
                print('spearmann_df_list[i]', spearmann_df_list[i])
                print('Available columns:', spearmann_df_list[i].columns.tolist())
                # wssd_values.append((spearmann_df_list[i][method_pc].iloc[0] + spearmann_df_list[i][method_rc].iloc[0]) / 2)
                # pred_values.append((pred_metrics_pc_list[i][method] + pred_metrics_rc_list[i][method]) / 2)
                wssd_values.append(spearmann_df_list[i][method_pc].iloc[0])
                wssd_values.append(spearmann_df_list[i][method_rc].iloc[0])
                pred_values.append(pred_metrics_pc_list[i][method])
                pred_values.append(pred_metrics_rc_list[i][method])
    # Calculate correlation
    if len(wssd_values) > 1 and len(pred_values) > 1:
        corr, _ = scipy.stats.pearsonr(wssd_values, pred_values)
        wssd_pred_corr["spearmann"] = corr
        corr_spearman, _ = scipy.stats.spearmanr(wssd_values, pred_values)
        wssd_pred_corr_spearman["spearmann"] = corr_spearman
    else:
        wssd_pred_corr["spearmann"] = None
        wssd_pred_corr_spearman["spearmann"] = None
    
    print("\nCorrelation between spearmann correlation and prediction metrics:")
    print(f"Spearmann correlation: Correlation = {wssd_pred_corr['spearmann']}")
    print(f"Spearmann correlation: Spearman Correlation = {wssd_pred_corr_spearman['spearmann']}")

    # now for kendall tau
    wssd_values = []
    pred_values = []
    for i in range(len(delta_list)):
        for method in pred_metrics_pc_list[i]:
            method_pc = " " + method + "-LSTM"
            method_rc = " " + method + "-MLP"
            if method_pc in kendall_df_list[i] and method_rc in kendall_df_list[i]:
                # wssd_values.append((kendall_df_list[i][method_pc].iloc[0] + kendall_df_list[i][method_rc].iloc[0]) / 2)
                # pred_values.append((pred_metrics_pc_list[i][method] + pred_metrics_rc_list[i][method]) / 2)
                wssd_values.append(kendall_df_list[i][method_pc].iloc[0])
                wssd_values.append(kendall_df_list[i][method_rc].iloc[0])
                pred_values.append(pred_metrics_pc_list[i][method])
                pred_values.append(pred_metrics_rc_list[i][method])
    # Calculate correlation
    if len(wssd_values) > 1 and len(pred_values) > 1:
        corr, _ = scipy.stats.pearsonr(wssd_values, pred_values)
        wssd_pred_corr["kendall"] = corr
        corr_spearman, _ = scipy.stats.spearmanr(wssd_values, pred_values)
        wssd_pred_corr_spearman["kendall"] = corr_spearman
    else:
        wssd_pred_corr["kendall"] = None
        wssd_pred_corr_spearman["kendall"] = None
    
    print("\nCorrelation between kendall tau correlation and prediction metrics:")
    print(f"Kendall tau correlation: Correlation = {wssd_pred_corr['kendall']}")
    print(f"Kendall tau correlation: Spearman Correlation = {wssd_pred_corr_spearman['kendall']}")


def plot_wssd_results(full_dict_wssd, full_dict_conf_int, intervention):
    policies = ["random", "S-Learner_LSTM", "T-Learner_LSTM", "TarNet_LSTM", "S-Learner_Vanilla_NN", "T-Learner_Vanilla_NN", "TarNet_Vanilla_NN"]

    x = np.arange(len(policies))  # x locations for policies
    width = 0.05  # Bar width
    gap = 0.015  # Gap between bars
    extra_gap = 0.05
    # transparency = 0.2  # Transparency for ProCause bars
    best_color = 'gold'

    color_dict = {"TarNet": "C0", "S-Learner": "C1", "T-Learner": "C2", "Ensemble": "C3"}
    # make hatch dict, with no hatch for MLP and // for LSTM
    hatch_dict = {"LSTM": "////", "MLP": ""}

    fig, ax = plt.subplots(figsize=(15, 5))

    # Loop through each policy to determine min values for transparency effect
    for i, policy in enumerate(policies):
        # Get values for current policy
        values = {
            'TarNet pro': full_dict_wssd["TarNet-LSTM"][policy],
            'S-Learner pro': full_dict_wssd["S-Learner-LSTM"][policy],
            'T-Learner pro': full_dict_wssd["T-Learner-LSTM"][policy],
            'TarNet real': full_dict_wssd["TarNet-MLP"][policy],
            'S-Learner real': full_dict_wssd["S-Learner-MLP"][policy],
            'T-Learner real': full_dict_wssd["T-Learner-MLP"][policy],
            'Ensemble ProCause': full_dict_wssd["Ensemble ProCause"][policy],
            'Ensemble RealCause': full_dict_wssd["Ensemble RealCause"][policy]
        }

        # Find the method with the lowest value
        min_method = min(values, key=values.get)

        ax.bar(i - 1.5 * width - gap, values['TarNet real'], width,
            label='RealCause TarNet' if i == 0 else "",
                edgecolor=best_color if min_method == 'TarNet real' else 'black',
                    linewidth=1.75, yerr=full_dict_conf_int["TarNet-MLP"][policy],
                        hatch=hatch_dict["MLP"], color=color_dict["TarNet"])
        
        ax.bar(i - 0.5 * width, values['TarNet pro'], width,
                label='ProCause TarNet' if i == 0 else "",
                    edgecolor=best_color if min_method == 'TarNet pro' else 'black',
                        linewidth=1.75, yerr=full_dict_conf_int["TarNet-LSTM"][policy],
                            hatch=hatch_dict["LSTM"], color=color_dict["TarNet"])
        

        
        ax.bar(i + 0.5 * width + gap + extra_gap, values['S-Learner real'], width,
                label='RealCause S-Learner' if i == 0 else "",
                    edgecolor=best_color if min_method == 'S-Learner real' else 'black',
                        linewidth=1.75, yerr=full_dict_conf_int["S-Learner-MLP"][policy]
                            , hatch=hatch_dict["MLP"], color=color_dict["S-Learner"])
        
        ax.bar(i + 1.5 * width + 2*gap + extra_gap, values['S-Learner pro'], width,
                    label='ProCause S-Learner' if i == 0 else "",
                        edgecolor=best_color if min_method == 'S-Learner pro' else 'black',
                            linewidth=1.75, yerr=full_dict_conf_int["S-Learner-LSTM"][policy]
                                , hatch=hatch_dict["LSTM"], color=color_dict["S-Learner"])
        
        

        ax.bar(i + 2.5 * width + 3*gap + 2*extra_gap, values['T-Learner real'], width,
                    label='RealCause T-Learner' if i == 0 else "",
                        edgecolor=best_color if min_method == 'T-Learner real' else 'black',
                            linewidth=1.75, yerr=full_dict_conf_int["T-Learner-MLP"][policy]
                                , hatch=hatch_dict["MLP"], color=color_dict["T-Learner"])
        
        ax.bar(i + 3.5 * width + 4*gap + 2*extra_gap, values['T-Learner pro'], width,
                    label='ProCause T-Learner' if i == 0 else "",
                        edgecolor=best_color if min_method == 'T-Learner pro' else 'black',
                            linewidth=1.75, yerr=full_dict_conf_int["T-Learner-LSTM"][policy]
                                , hatch=hatch_dict["LSTM"], color=color_dict["T-Learner"])
        


        ax.bar(i + 4.5 * width + 5*gap + 3*extra_gap, values['Ensemble RealCause'], width,
                    label='Ensemble RealCause' if i == 0 else "",
                        edgecolor=best_color if min_method == 'Ensemble RealCause' else 'black',
                            linewidth=1.75, yerr=full_dict_conf_int["Ensemble RealCause"][policy]
                                , hatch=hatch_dict["MLP"], color=color_dict["Ensemble"])
        
        ax.bar(i + 5.5 * width + 6*gap + 3*extra_gap, values['Ensemble ProCause'], width,
                    label='Ensemble ProCause' if i == 0 else "",
                        edgecolor=best_color if min_method == 'Ensemble ProCause' else 'black',
                            linewidth=1.75, yerr=full_dict_conf_int["Ensemble ProCause"][policy]
                                , hatch=hatch_dict["LSTM"], color=color_dict["Ensemble"])

    ax.set_ylabel('WSSD')
    # put title a bit higher than normal
    ax.set_title(f"WSSD per policy for intervention {intervention} ($\delta = 0.95$)", pad=60)
    ax.set_xticks(x)
    # delete the '_' in the policy names
    policies_no_ = [policy.replace('_', ' ') for policy in policies if policy != "bank"]
    # replace 'Vanilla NN' with 'MLP' for better visualization
    policies_no_vanilla = [policy.replace('Vanilla NN', 'MLP') for policy in policies_no_]
    # put the names of the policies on the x-axis, and put the 'second' word of the policy on the next line if it is present
    ax.set_xticklabels(policies_no_vanilla)
    # ax.set_xticklabels(policies, rotation=45)

    legend_handles = [
        mpatches.Patch(facecolor=color_dict["TarNet"], label='TarNet-MLP (RealCause)', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["TarNet"], label='TarNet-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["S-Learner"], label='S-Learner-MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["S-Learner"], label='S-Learner-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["T-Learner"], label='T-Learner-MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["T-Learner"], label='T-Learner-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["Ensemble"], label='Ensemble MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["Ensemble"], label='Ensemble LSTM', hatch=hatch_dict["LSTM"])
    ]

    # put horizontal legend at top of the plot
    ax.legend(handles=legend_handles, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))

    fig.tight_layout()
    plt.show()

    # make another plot, where you average over all policies (random, S-Learner, T-Learner, TarNet)
    fig, ax = plt.subplots(figsize=(15, 5))

    # Loop through each policy to determine min values for transparency effect

    values = {
        'TarNet pro': np.mean([full_dict_wssd["TarNet-LSTM"][policy] for policy in policies if policy != "bank"]),
        'S-Learner pro': np.mean([full_dict_wssd["S-Learner-LSTM"][policy] for policy in policies if policy != "bank"]),
        'T-Learner pro': np.mean([full_dict_wssd["T-Learner-LSTM"][policy] for policy in policies if policy != "bank"]),
        'TarNet real': np.mean([full_dict_wssd["TarNet-MLP"][policy] for policy in policies if policy != "bank"]),
        'S-Learner real': np.mean([full_dict_wssd["S-Learner-MLP"][policy] for policy in policies if policy != "bank"]),
        'T-Learner real': np.mean([full_dict_wssd["T-Learner-MLP"][policy] for policy in policies if policy != "bank"]),
        'Ensemble ProCause': np.mean([full_dict_wssd["Ensemble ProCause"][policy] for policy in policies if policy != "bank"]),
        'Ensemble RealCause': np.mean([full_dict_wssd["Ensemble RealCause"][policy] for policy in policies if policy != "bank"])
    }

    # get standard error by dividing by 1.96, square the standard errors, sum them up, and divide, and then take the square root
    standard_error_values = {
        'TarNet pro': np.sqrt(np.sum([ (full_dict_conf_int["TarNet-LSTM"][policy] / 1.96) ** 2 for policy in policies if policy != "bank"]) / len(policies)),
        'S-Learner pro': np.sqrt(np.sum([ (full_dict_conf_int["S-Learner-LSTM"][policy] / 1.96) ** 2 for policy in policies if policy != "bank"]) / len(policies)),
        'T-Learner pro': np.sqrt(np.sum([ (full_dict_conf_int["T-Learner-LSTM"][policy] / 1.96) ** 2 for policy in policies if policy != "bank"]) / len(policies)),
        'TarNet real': np.sqrt(np.sum([ (full_dict_conf_int["TarNet-MLP"][policy] / 1.96) ** 2 for policy in policies if policy != "bank"]) / len(policies)),
        'S-Learner real': np.sqrt(np.sum([ (full_dict_conf_int["S-Learner-MLP"][policy] / 1.96) ** 2 for policy in policies if policy != "bank"]) / len(policies)),
        'T-Learner real': np.sqrt(np.sum([ (full_dict_conf_int["T-Learner-MLP"][policy] / 1.96) ** 2 for policy in policies if policy != "bank"]) / len(policies)),
        'Ensemble ProCause': np.sqrt(np.sum([ (full_dict_conf_int["Ensemble ProCause"][policy] / 1.96) ** 2 for policy in policies if policy != "bank"]) / len(policies)),
        'Ensemble RealCause': np.sqrt(np.sum([ (full_dict_conf_int["Ensemble RealCause"][policy] / 1.96) ** 2 for policy in policies if policy != "bank"]) / len(policies))
    }

    # Find the method with the lowest value
    min_method = min(values, key=values.get)

    ax.bar(-1.5 * width - gap, values['TarNet real'], width,
        label='RealCause TarNet',
            edgecolor=best_color if min_method == 'TarNet real' else 'black',
                linewidth=1.75, yerr=np.mean([full_dict_conf_int["TarNet-MLP"][policy] for policy in policies if policy != "bank"]),
                    hatch=hatch_dict["MLP"], color=color_dict["TarNet"])
    
    ax.bar(-0.5 * width, values['TarNet pro'], width,
            label='ProCause TarNet',
                edgecolor=best_color if min_method == 'TarNet pro' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["TarNet-LSTM"][policy] for policy in policies if policy != "bank"]),
                        hatch=hatch_dict["LSTM"], color=color_dict["TarNet"])
    
    
    ax.bar(0.5 * width + gap + extra_gap, values['S-Learner real'], width,
            label='RealCause S-Learner',
                edgecolor=best_color if min_method == 'S-Learner real' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["S-Learner-MLP"][policy] for policy in policies if policy != "bank"]),
                        hatch=hatch_dict["MLP"], color=color_dict["S-Learner"])
    
    ax.bar(1.5 * width + 2*gap + extra_gap, values['S-Learner pro'], width,
                label='ProCause S-Learner',
                    edgecolor=best_color if min_method == 'S-Learner pro' else 'black',
                        linewidth=1.75, yerr=np.mean([full_dict_conf_int["S-Learner-LSTM"][policy] for policy in policies if policy != "bank"])
                            , hatch=hatch_dict["LSTM"], color=color_dict["S-Learner"])
    

    ax.bar(2.5 * width + 3*gap + 2*extra_gap, values['T-Learner real'], width,
                label='RealCause T-Learner',
                    edgecolor=best_color if min_method == 'T-Learner real' else 'black',
                        linewidth=1.75, yerr=np.mean([full_dict_conf_int["T-Learner-MLP"][policy] for policy in policies if policy != "bank"])
                            , hatch=hatch_dict["MLP"], color=color_dict["T-Learner"])
    
    ax.bar(3.5 * width + 4*gap + 2*extra_gap, values['T-Learner pro'], width,
                label='ProCause T-Learner',
                    edgecolor=best_color if min_method == 'T-Learner pro' else 'black',
                        linewidth=1.75, yerr=np.mean([full_dict_conf_int["T-Learner-LSTM"][policy] for policy in policies if policy != "bank"])
                            , hatch=hatch_dict["LSTM"], color=color_dict["T-Learner"])
    

    ax.bar(4.5 * width + 5*gap + 3*extra_gap, values['Ensemble RealCause'], width,
                label='Ensemble RealCause',
                    edgecolor=best_color if min_method == 'Ensemble RealCause' else 'black',
                        linewidth=1.75, yerr=np.mean([full_dict_conf_int["Ensemble RealCause"][policy] for policy in policies if policy != "bank"])
                            , hatch=hatch_dict["MLP"], color=color_dict["Ensemble"])
    
    ax.bar(5.5 * width + 6*gap + 3*extra_gap, values['Ensemble ProCause'], width,
                label='Ensemble ProCause',
                    edgecolor=best_color if min_method == 'Ensemble ProCause' else 'black',
                        linewidth=1.75, yerr=np.mean([full_dict_conf_int["Ensemble ProCause"][policy] for policy in policies if policy != "bank"])
                            , hatch=hatch_dict["LSTM"], color=color_dict["Ensemble"])
    
    print('Values for all policies:', values)
    print('Standard errors:', standard_error_values)

    ax.set_ylabel('WSSD')
    # put title a bit higher than normal
    ax.set_title(f"WSSD averaged over all policies for intervention {intervention} ($\delta = 0.95$)", pad=60)
    # ax.set_xticks([-1, 0, 1, 2, 3, 4, 5, 6])
    # ax.set_xticklabels(['TarNet-MLP', 'TarNet-LSTM', 'S-Learner-MLP', 'S-Learner-LSTM', 'T-Learner-MLP', 'T-Learner-LSTM', 'Ensemble MLP', 'Ensemble LSTM'])
    # ax.set_xticklabels(policies, rotation=45)

    legend_handles = [
        mpatches.Patch(facecolor=color_dict["TarNet"], label='TarNet-MLP (RealCause)', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["TarNet"], label='TarNet-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["S-Learner"], label='S-Learner-MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["S-Learner"], label='S-Learner-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["T-Learner"], label='T-Learner-MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["T-Learner"], label='T-Learner-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["Ensemble"], label='Ensemble MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["Ensemble"], label='Ensemble LSTM', hatch=hatch_dict["LSTM"])
    ]

    # put horizontal legend at top of the plot
    ax.legend(handles=legend_handles, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))

    fig.tight_layout()
    plt.show()

    # now do the same thing but for all policies except the random policy
    fig, ax = plt.subplots(figsize=(15, 5))

    # Loop through each policy to determine min values for transparency effect

    values = {
        'TarNet pro': np.mean([full_dict_wssd["TarNet-LSTM"][policy] for policy in policies if policy != "bank" and policy != "random"]),
        'S-Learner pro': np.mean([full_dict_wssd["S-Learner-LSTM"][policy] for policy in policies if policy != "bank" and policy != "random"]),
        'T-Learner pro': np.mean([full_dict_wssd["T-Learner-LSTM"][policy] for policy in policies if policy != "bank" and policy != "random"]),
        'TarNet real': np.mean([full_dict_wssd["TarNet-MLP"][policy] for policy in policies if policy != "bank" and policy != "random"]),
        'S-Learner real': np.mean([full_dict_wssd["S-Learner-MLP"][policy] for policy in policies if policy != "bank" and policy != "random"]),
        'T-Learner real': np.mean([full_dict_wssd["T-Learner-MLP"][policy] for policy in policies if policy != "bank" and policy != "random"]),
        'Ensemble ProCause': np.mean([full_dict_wssd["Ensemble ProCause"][policy] for policy in policies if policy != "bank" and policy != "random"]),
        'Ensemble RealCause': np.mean([full_dict_wssd["Ensemble RealCause"][policy] for policy in policies if policy != "bank" and policy != "random"])
    }

    # get standard error by dividing by 1.96, square the standard errors, sum them up, and divide, and then take the square root
    standard_error_values = {
        'TarNet pro': np.sqrt(np.sum([ (full_dict_conf_int["TarNet-LSTM"][policy] / 1.96) ** 2 for policy in policies if policy != "bank" and policy != "random"]) / len(policies[1:])),
        'S-Learner pro': np.sqrt(np.sum([ (full_dict_conf_int["S-Learner-LSTM"][policy] / 1.96) ** 2 for policy in policies if policy != "bank" and policy != "random"]) / len(policies[1:])),
        'T-Learner pro': np.sqrt(np.sum([ (full_dict_conf_int["T-Learner-LSTM"][policy] / 1.96) ** 2 for policy in policies if policy != "bank" and policy != "random"]) / len(policies[1:])),
        'TarNet real': np.sqrt(np.sum([ (full_dict_conf_int["TarNet-MLP"][policy] / 1.96) ** 2 for policy in policies if policy != "bank" and policy != "random"]) / len(policies[1:])),
        'S-Learner real': np.sqrt(np.sum([ (full_dict_conf_int["S-Learner-MLP"][policy] / 1.96) ** 2 for policy in policies if policy != "bank" and policy != "random"]) / len(policies[1:])),
        'T-Learner real': np.sqrt(np.sum([ (full_dict_conf_int["T-Learner-MLP"][policy] / 1.96) ** 2 for policy in policies if policy != "bank" and policy != "random"]) / len(policies[1:])),
        'Ensemble ProCause': np.sqrt(np.sum([ (full_dict_conf_int["Ensemble ProCause"][policy] / 1.96) ** 2 for policy in policies if policy != "bank" and policy != "random"]) / len(policies[1:])),
        'Ensemble RealCause': np.sqrt(np.sum([ (full_dict_conf_int["Ensemble RealCause"][policy] / 1.96) ** 2 for policy in policies if policy != "bank" and policy != "random"]) / len(policies[1:]))
    }

    # Find the method with the lowest value
    min_method = min(values, key=values.get)

    ax.bar(-1.5 * width - gap, values['TarNet real'], width,
        label='RealCause TarNet',
            edgecolor=best_color if min_method == 'TarNet real' else 'black',
                linewidth=1.75, yerr=np.mean([full_dict_conf_int["TarNet-MLP"][policy] for policy in policies if policy != "bank"]),
                    hatch=hatch_dict["MLP"], color=color_dict["TarNet"])
    
    ax.bar(-0.5 * width, values['TarNet pro'], width,
            label='ProCause TarNet',
                edgecolor=best_color if min_method == 'TarNet pro' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["TarNet-LSTM"][policy] for policy in policies if policy != "bank"]),
                        hatch=hatch_dict["LSTM"], color=color_dict["TarNet"])
    

    ax.bar(0.5 * width + gap + extra_gap, values['S-Learner real'], width,
            label='RealCause S-Learner',
                edgecolor=best_color if min_method == 'S-Learner real' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["S-Learner-MLP"][policy] for policy in policies if policy != "bank"])
                        , hatch=hatch_dict["MLP"], color=color_dict["S-Learner"])
    
    ax.bar(1.5 * width + 2*gap + extra_gap, values['S-Learner pro'], width,
            label='ProCause S-Learner',
                edgecolor=best_color if min_method == 'S-Learner pro' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["S-Learner-LSTM"][policy] for policy in policies if policy != "bank"])
                        , hatch=hatch_dict["LSTM"], color=color_dict["S-Learner"])
    

    ax.bar(2.5 * width + 3*gap + 2*extra_gap, values['T-Learner real'], width,
            label='RealCause T-Learner',
                edgecolor=best_color if min_method == 'T-Learner real' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["T-Learner-MLP"][policy] for policy in policies if policy != "bank"])
                        , hatch=hatch_dict["MLP"], color=color_dict["T-Learner"])
    
    ax.bar(3.5 * width + 4*gap + 2*extra_gap, values['T-Learner pro'], width,
            label='ProCause T-Learner',
                edgecolor=best_color if min_method == 'T-Learner pro' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["T-Learner-LSTM"][policy] for policy in policies if policy != "bank"])
                        , hatch=hatch_dict["LSTM"], color=color_dict["T-Learner"])
    

    ax.bar(4.5 * width + 5*gap + 3*extra_gap, values['Ensemble RealCause'], width,
            label='Ensemble RealCause',
                edgecolor=best_color if min_method == 'Ensemble RealCause' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["Ensemble RealCause"][policy] for policy in policies if policy != "bank"])
                        , hatch=hatch_dict["MLP"], color=color_dict["Ensemble"])
    
    ax.bar(5.5 * width + 6*gap + 3*extra_gap, values['Ensemble ProCause'], width,
            label='Ensemble ProCause',
                edgecolor=best_color if min_method == 'Ensemble ProCause' else 'black',
                    linewidth=1.75, yerr=np.mean([full_dict_conf_int["Ensemble ProCause"][policy] for policy in policies if policy != "bank"])
                        , hatch=hatch_dict["LSTM"], color=color_dict["Ensemble"])
    
    print('Values for non-random policies:', values)
    print('Standard errors:', standard_error_values)

    ax.set_ylabel('WSSD')
    # put title a bit higher than normal
    ax.set_title(f"WSSD averaged over all non-random policies for intervention {intervention} ($\delta = 0.95$)", pad=60)
    # ax.set_xticks([-1, 0, 1, 2, 3, 4, 5, 6])
    # ax.set_xticklabels(['TarNet-MLP', 'TarNet-LSTM', 'S-Learner-MLP', 'S-Learner-LSTM', 'T-Learner-MLP', 'T-Learner-LSTM', 'Ensemble MLP', 'Ensemble LSTM'])
    # ax.set_xticklabels(policies, rotation=45)

    legend_handles = [
        mpatches.Patch(facecolor=color_dict["TarNet"], label='TarNet-MLP (RealCause)', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["TarNet"], label='TarNet-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["S-Learner"], label='S-Learner-MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["S-Learner"], label='S-Learner-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["T-Learner"], label='T-Learner-MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["T-Learner"], label='T-Learner-LSTM', hatch=hatch_dict["LSTM"]),
        mpatches.Patch(facecolor=color_dict["Ensemble"], label='Ensemble MLP', hatch=hatch_dict["MLP"]),
        mpatches.Patch(facecolor=color_dict["Ensemble"], label='Ensemble LSTM', hatch=hatch_dict["LSTM"])
    ]

    # put horizontal legend at top of the plot
    ax.legend(handles=legend_handles, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))

    fig.tight_layout()
    plt.show()

    return fig, ax

def plot_ensemble_comparison(intervention, tsne=False, distr=False, sort_by='TarNet'):
    delta = 0.95
    policies = str(["all"])
    if distr:

        # ensemble_random_wssd_pc = load_data(path + "\\res\\SimBank\\ensemble_random_wssd_pc" + intervention + ".pkl")
        # ensemble_random_wssd_rc = load_data(path + "\\res\\SimBank\\ensemble_random_wssd_rc" + intervention + ".pkl")
        # tarnet_random_wssd_pc = load_data(path + "\\res\\SimBank\\tarnet_random_wssd_pc" + intervention + ".pkl")
        # tarnet_random_wssd_rc = load_data(path + "\\res\\SimBank\\tarnet_random_wssd_rc" + intervention + ".pkl")
        # slearner_random_wssd_pc = load_data(path + "\\res\\SimBank\\slearner_random_wssd_pc" + intervention + ".pkl")
        # slearner_random_wssd_rc = load_data(path + "\\res\\SimBank\\slearner_random_wssd_rc" + intervention + ".pkl")
        # tlearner_random_wssd_pc = load_data(path + "\\res\\SimBank\\tlearner_random_wssd_pc" + intervention + ".pkl")
        # tlearner_random_wssd_rc = load_data(path + "\\res\\SimBank\\tlearner_random_wssd_rc" + intervention + ".pkl")
        ensemble_random_wssd_pc = load_data(path + "\\res\\SimBank\\ensemble_random_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
        ensemble_random_wssd_rc = load_data(path + "\\res\\SimBank\\ensemble_random_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
        tarnet_random_wssd_pc = load_data(path + "\\res\\SimBank\\tarnet_random_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
        tarnet_random_wssd_rc = load_data(path + "\\res\\SimBank\\tarnet_random_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
        slearner_random_wssd_pc = load_data(path + "\\res\\SimBank\\slearner_random_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
        slearner_random_wssd_rc = load_data(path + "\\res\\SimBank\\slearner_random_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
        tlearner_random_wssd_pc = load_data(path + "\\res\\SimBank\\tlearner_random_wssd_pc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")
        tlearner_random_wssd_rc = load_data(path + "\\res\\SimBank\\tlearner_random_wssd_rc" + intervention + str(delta) + str(policies) + fixed_policy_delta_to_add + ".pkl")

        # discard all the perfect 0s
        ensemble_random_wssd_pc = [x for x in ensemble_random_wssd_pc if x > 0]
        tarnet_random_wssd_pc = [x for x in tarnet_random_wssd_pc if x > 0]
        slearner_random_wssd_pc = [x for x in slearner_random_wssd_pc if x > 0]
        tlearner_random_wssd_pc = [x for x in tlearner_random_wssd_pc if x > 0]
        ensemble_random_wssd_rc = [x for x in ensemble_random_wssd_rc if x > 0]
        tarnet_random_wssd_rc = [x for x in tarnet_random_wssd_rc if x > 0]
        slearner_random_wssd_rc = [x for x in slearner_random_wssd_rc if x > 0]
        tlearner_random_wssd_rc = [x for x in tlearner_random_wssd_rc if x > 0]

        print('len of ensemble_random_wssd_pc:', len(ensemble_random_wssd_pc))
        print('len of tarnet_random_wssd_pc:', len(tarnet_random_wssd_pc))
        print('len of slearner_random_wssd_pc:', len(slearner_random_wssd_pc))
        print('len of tlearner_random_wssd_pc:', len(tlearner_random_wssd_pc))
        print('len of ensemble_random_wssd_rc:', len(ensemble_random_wssd_rc))
        print('len of tarnet_random_wssd_rc:', len(tarnet_random_wssd_rc))
        print('len of slearner_random_wssd_rc:', len(slearner_random_wssd_rc))
        print('len of tlearner_random_wssd_rc:', len(tlearner_random_wssd_rc))

        color_map = {
            "tarnet": "#1f77b4",      # blue
            "s-learner": "#2ca02c",   # green
            "t-learner": "#ff7f0e",   # orange
            "ensemble": "#d62728",    # red
            "mlp": "#000000",         # black
            "lstm": "#7f00ff",        # dark violet
        }
        from scipy.stats import gaussian_kde
        from matplotlib.patches import Patch

        def plot_kde_line(data, color, label):
            kde = gaussian_kde(data)
            x_vals = np.linspace(min(data), max(data), 1000)
            y_vals = kde(x_vals)
            plt.plot(x_vals, y_vals, color=color, linewidth=2.5, label=label)

        plt.figure(figsize=(8, 4))
        bins = 150
        alpha1 = 0.2
        alpha2 = 0.25
        density = False

        # Histogram plots
        plt.hist(tarnet_random_wssd_pc, bins=bins, color=color_map["tarnet"], alpha=alpha1, density=True)
        plt.hist(slearner_random_wssd_pc, bins=bins, color=color_map["s-learner"], alpha=alpha1, density=True)
        plt.hist(tlearner_random_wssd_pc, bins=bins, color=color_map["t-learner"], alpha=alpha1, density=True)
        plt.hist(ensemble_random_wssd_pc, bins=bins, color=color_map["ensemble"], alpha=alpha2, edgecolor='k', density=True)

        # KDE lines (thicker and separate)
        plot_kde_line(tarnet_random_wssd_pc, color_map["tarnet"], "TARNet")
        plot_kde_line(slearner_random_wssd_pc, color_map["s-learner"], "S-learner")
        plot_kde_line(tlearner_random_wssd_pc, color_map["t-learner"], "T-learner")
        plot_kde_line(ensemble_random_wssd_pc, color_map["ensemble"], "Ensemble")

        # Legend (all elements alpha=1)
        legend_elements = [
            Patch(facecolor=color_map["tarnet"], label="TARNet"),
            Patch(facecolor=color_map["s-learner"], label="S-learner"),
            Patch(facecolor=color_map["t-learner"], label="T-learner"),
            Patch(facecolor=color_map["ensemble"], edgecolor='k', label="Ensemble")
        ]
        plt.legend(handles=legend_elements, fontsize=14)

        plt.title("Probability density of WD: Individual Models vs Ensemble for evaluating\n the $\\mathit{random}$ policy"
          " (intervention $\\mathit{Time\\ Contact\\ HQ}$; $\delta=0.95$)", fontsize=16)
        plt.xlabel("WD")
        plt.ylabel("Density")
        plt.tight_layout()

        plt.locator_params(axis='x', nbins=4)
        plt.locator_params(axis='y', nbins=4)

        # save the figure as pdf
        plt.savefig(path + "\\res\\SimBank\\ensemble_comparison_" + intervention + ".pdf", bbox_inches='tight')

        plt.show()



        # plot an error distribution plot for the ensemble and the individual methods for rc and pc separately
        # Plot individual model distributions
        # plt.figure(figsize=(8, 6))
        # bins = 150
        # alpha1 = 0.2
        # alpha2 = 0.5


        # sns.histplot(tarnet_random_wssd_pc, bins=bins, color=color_map["tarnet"], alpha=alpha1, label="TARNet", kde=True)
        # sns.histplot(slearner_random_wssd_pc, bins=bins, color=color_map["s-learner"], alpha=alpha1, label="S-learner", kde=True)
        # sns.histplot(tlearner_random_wssd_pc, bins=bins, color=color_map["t-learner"], alpha=alpha1, label="T-learner", kde=True)
        # # Plot ensemble
        # sns.histplot(ensemble_random_wssd_pc, bins=bins, color=color_map["ensemble"], alpha=alpha2, label="Ensemble", kde=True, edgecolor='k')
        # plt.legend()
        # plt.title("WD Distribution: Individual Models vs Ensemble")
        # plt.xlabel("WD")
        # plt.ylabel("Frequency")
        # plt.show()

        plt.figure(figsize=(8, 6))
        sns.kdeplot(tarnet_random_wssd_pc, color=color_map["tarnet"], label="TARNET", linewidth=2, clip=(0, max(tarnet_random_wssd_pc)))
        sns.kdeplot(slearner_random_wssd_pc, color=color_map["s-learner"], label="S-learner", linewidth=2, clip=(0, max(slearner_random_wssd_pc)))
        sns.kdeplot(tlearner_random_wssd_pc, color=color_map["t-learner"], label="T-learner", linewidth=2, clip=(0, max(tlearner_random_wssd_pc)))
        sns.kdeplot(ensemble_random_wssd_pc, color=color_map["ensemble"], label="Ensemble", linewidth=3, clip=(0, max(ensemble_random_wssd_pc)))
        # Vertical lines for mean errors
        plt.axvline(np.mean(tarnet_random_wssd_pc), color=color_map["tarnet"], linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(slearner_random_wssd_pc), color=color_map["s-learner"], linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(tlearner_random_wssd_pc), color=color_map["t-learner"], linestyle='dashed', linewidth=1)
        plt.axvline(np.mean(ensemble_random_wssd_pc), color=color_map["ensemble"], linestyle='dashed', linewidth=1)
        plt.legend()
        plt.title("WD Distribution: Individual Models vs Ensemble")
        plt.xlabel("WD Error")
        plt.ylabel("Frequency")
        plt.show()

        plt.figure(figsize=(8, 6))
        sns.ecdfplot(tarnet_random_wssd_pc, color=color_map["tarnet"], label="TARNET", linewidth=2)
        sns.ecdfplot(slearner_random_wssd_pc, color=color_map["s-learner"], label="S-learner", linewidth=2)
        sns.ecdfplot(tlearner_random_wssd_pc, color=color_map["t-learner"], label="T-learner", linewidth=2)
        sns.ecdfplot(ensemble_random_wssd_pc, color=color_map["ensemble"], label="Ensemble", linewidth=3, linestyle='--')
        plt.title("WD CDF: Ensemble vs Individual Models", fontsize=14)
        plt.xlabel("WD", fontsize=12)
        plt.ylabel("Cumulative Probability", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.show()

        # Sort the errors for TarNet in ascending order, and then plot them (with the same order as the TarNet one for the other methods, so remember the order in some way
        if sort_by == 'TarNet':
        # tarnet_random_wssd_pc_sorted = np.sort(tarnet_random_wssd_pc)
            pc_order = np.argsort(tarnet_random_wssd_pc)
        elif sort_by == 'S-Learner':
            pc_order = np.argsort(slearner_random_wssd_pc)
        elif sort_by == 'T-Learner':
            pc_order = np.argsort(tlearner_random_wssd_pc)
        elif sort_by == 'Ensemble':
            pc_order = np.argsort(ensemble_random_wssd_pc)
        tarnet_random_wssd_pc_sorted = np.array(tarnet_random_wssd_pc)[pc_order]
        slearner_random_wssd_pc_sorted = np.array(slearner_random_wssd_pc)[pc_order]
        tlearner_random_wssd_pc_sorted = np.array(tlearner_random_wssd_pc)[pc_order]
        ensemble_random_wssd_pc_sorted = np.array(ensemble_random_wssd_pc)[pc_order]
        # plot separately
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(tarnet_random_wssd_pc_sorted)), tarnet_random_wssd_pc_sorted, label="TARNET", color=color_map["tarnet"], alpha=0.5, linestyle='-', marker='o', markersize=2)
        plt.title("WSSD Error Sorted: TARNet")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(slearner_random_wssd_pc_sorted)), slearner_random_wssd_pc_sorted, label="S-learner", color=color_map["s-learner"], alpha=0.5, linestyle='-', marker='o', markersize=2)
        plt.title("WSSD Error Sorted: S-Learner")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(tlearner_random_wssd_pc_sorted)), tlearner_random_wssd_pc_sorted, label="T-learner", color=color_map["t-learner"], alpha=0.5, linestyle='-', marker='o', markersize=2)
        plt.title("WSSD Error Sorted: T-Learner")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(ensemble_random_wssd_pc_sorted)), ensemble_random_wssd_pc_sorted, label="Ensemble", color=color_map["ensemble"], alpha=0.5, linestyle='-', marker='o', markersize=2)
        plt.title("WSSD Error Sorted: Ensemble")
        plt.show()

        # Combined plot with moving average and real points
        window_size = 10  # Define the window size for the moving average
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        plt.figure(figsize=(10, 8))
        plt.plot(moving_average(tarnet_random_wssd_pc_sorted, window_size), label="TARNET (Moving Avg)", color=color_map["tarnet"], alpha=0.8, linestyle='-', marker='o', markersize=2)
        plt.plot(moving_average(slearner_random_wssd_pc_sorted, window_size), label="S-learner (Moving Avg)", color=color_map["s-learner"], alpha=0.8, linestyle='-', marker='o', markersize=2)
        plt.plot(moving_average(tlearner_random_wssd_pc_sorted, window_size), label="T-learner (Moving Avg)", color=color_map["t-learner"], alpha=0.8, linestyle='-', marker='o', markersize=2)
        plt.plot(moving_average(ensemble_random_wssd_pc_sorted, window_size), label="Ensemble (Moving Avg)", color=color_map["ensemble"], alpha=0.8, linestyle='-', marker='o', markersize=2)
        plt.title("WSSD Error Sorted: Combined (Moving Average)")
        plt.xlabel("Index")
        plt.ylabel("WSSD Error")
        plt.legend()
        plt.show()

    # Get test dataset, do TSNE on it, and plot the TSNE plot
    if tsne:
        import torch
        prep_agg_offline_full_dict = load_data(os.path.join(os.getcwd(), "SimBank", "prep_agg_offline_full_dict_" + str(intervention)) + "0.95" + "_generator_iteration" + str(0) + "RealCause")
        # just take the first iteration
        X = prep_agg_offline_full_dict["random"][0]["X"][: , :-1]
        case_nr_X = prep_agg_offline_full_dict["random"][0]["X"][:, -1]
        offline_actions_dfs = load_data(os.path.join(os.getcwd(), "SimBank", "offline_actions_dfs_" + str(intervention) + "0.95" + "_generator_iteration" + str(0) + "RealCause"))
        T = np.array(offline_actions_dfs["random"][0]["action"].values, dtype=np.int64)
        # convert case_nr_X to array of integers
        case_nr_X = np.array([int(x) for x in case_nr_X])
        # T is a tensor with values like [0, 1, 0], so let's multiply it by the values of the treatment
        if intervention == "['set_ir_3_levels']":
            levels = np.array([0.07, 0.08, 0.09], dtype=np.float32)
            # T = levels[T]
        else:
            levels = ["skip_contact", "contact_headquarters"]


        # get the ensemble predictions, and the individual predictions, and the true labels
        tarnet_outcomes_pc = load_data(os.path.join(os.getcwd(), "SimBank", "estimated_outcome_dfs_procause_" + str(intervention) + "0.95" + "_generator_iteration" + str(0) + "ProCauseTarNet"))
        tarnet_outcomes_pc = tarnet_outcomes_pc["random"][0]
        slearner_outcomes_pc = load_data(os.path.join(os.getcwd(), "SimBank", "estimated_outcome_dfs_procause_" + str(intervention) + "0.95" + "_generator_iteration" + str(0) + "ProCauseS-Learner"))
        slearner_outcomes_pc = slearner_outcomes_pc["random"][0]
        tlearner_outcomes_pc = load_data(os.path.join(os.getcwd(), "SimBank", "estimated_outcome_dfs_procause_" + str(intervention) + "0.95" + "_generator_iteration" + str(0) + "ProCauseT-Learner"))
        tlearner_outcomes_pc = tlearner_outcomes_pc["random"][0]

        ensemble_pc = deepcopy(tarnet_outcomes_pc)
        ensemble_pc["outcome"] += deepcopy(slearner_outcomes_pc["outcome"]) + deepcopy(tlearner_outcomes_pc["outcome"])
        ensemble_pc["outcome"] = ensemble_pc["outcome"] / 3
        
        # get the true labels
        # true_outcome_dfs_path = "\\res\\SimBank\\online_outcome_dfs_" + intervention + "0.95" + "_generator_iteration" + str(0) + "RealCause"
        true_outcomes = load_data(os.path.join(os.getcwd(), "SimBank", "online_outcome_dfs_" + intervention + "0.95" + "_generator_iteration" + str(0) + "RealCause"))
        true_outcomes = true_outcomes["random"][0]

        # go through the tarnet_outcomes_pc, slearner_outcomes_pc, tlearner_outcomes_pc, ensemble_pc and take the average for every case_nr
        tarnet_outcomes_pc = tarnet_outcomes_pc.groupby("case_nr").mean()
        slearner_outcomes_pc = slearner_outcomes_pc.groupby("case_nr").mean()
        tlearner_outcomes_pc = tlearner_outcomes_pc.groupby("case_nr").mean()
        ensemble_pc = ensemble_pc.groupby("case_nr").mean()
        true_outcomes = true_outcomes.groupby("case_nr").mean()

        test = deepcopy(tarnet_outcomes_pc)
        test["outcome"] += deepcopy(slearner_outcomes_pc["outcome"]) + deepcopy(tlearner_outcomes_pc["outcome"])
        test["outcome"] = test["outcome"] / 3

        # get every df in the same order as case_nr_X (note that case_nr_X is not the same size though)
        tarnet_outcomes_pc = tarnet_outcomes_pc.loc[case_nr_X]
        slearner_outcomes_pc = slearner_outcomes_pc.loc[case_nr_X]
        tlearner_outcomes_pc = tlearner_outcomes_pc.loc[case_nr_X]
        ensemble_pc = ensemble_pc.loc[case_nr_X]
        # also make sure that true outcomes only contains the case_nrs that are in case_nr_X
        true_outcomes = true_outcomes.loc[case_nr_X]

        # do the tsne on X
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=1, random_state=42)
        X_embedded = tsne.fit_transform(X)

        # plot the tsne plot
        plt.figure(figsize=(8, 6))
        plt.scatter(X_embedded, tarnet_outcomes_pc["outcome"], label="TARNET", color='red', alpha=0.5)
        plt.scatter(X_embedded, slearner_outcomes_pc["outcome"], label="S-learner", color='#004d00', alpha=0.5)
        plt.scatter(X_embedded, tlearner_outcomes_pc["outcome"], label="T-learner", color='purple', alpha=0.5)
        plt.scatter(X_embedded, ensemble_pc["outcome"], label="Ensemble", color='blue', alpha=0.5)
        plt.scatter(X_embedded, true_outcomes["outcome"], label="True", color='black', alpha=0.5)
        plt.title("TSNE plot of the estimated outcomes vs the true outcomes")
        plt.xlabel("TSNE")
        plt.ylabel("Outcome")
        plt.legend()
        plt.show()
        print('lol')

        # plot the on the y-axis the true outcomes and on the x-axis the estimated outcomes, add a line for the true outcomes, and add lines for the estimated outcomes
        plt.figure(figsize=(8, 6))
        # plt.scatter(tarnet_outcomes_pc["outcome"], true_outcomes["outcome"], label="TARNET", color='red', alpha=0.5)
        # plt.scatter(slearner_outcomes_pc["outcome"], true_outcomes["outcome"], label="S-learner", color='#004d00', alpha=0.5)
        # plt.scatter(tlearner_outcomes_pc["outcome"], true_outcomes["outcome"], label="T-learner", color='purple', alpha=0.5)
        # plt.scatter(ensemble_pc["outcome"], true_outcomes["outcome"], label="Ensemble", color='blue', alpha=0.5)
        
        # Add a line for the true outcomes (y = x line)
        x_values = np.linspace(min(true_outcomes["outcome"]), max(true_outcomes["outcome"]), 10)
        plt.plot(x_values, x_values, label="True Outcomes", color='black', linestyle='dashed')
        # Add lines for the estimated outcomes
        plt.plot(np.sort(tarnet_outcomes_pc["outcome"]), np.sort(true_outcomes["outcome"]), color='red', linestyle='solid', alpha=0.3)
        plt.plot(np.sort(slearner_outcomes_pc["outcome"]), np.sort(true_outcomes["outcome"]), color='#004d00', linestyle='solid', alpha=0.3)
        plt.plot(np.sort(tlearner_outcomes_pc["outcome"]), np.sort(true_outcomes["outcome"]), color='purple', linestyle='solid', alpha=0.3)
        plt.plot(np.sort(ensemble_pc["outcome"]), np.sort(true_outcomes["outcome"]), color='blue', linestyle='solid', alpha=0.3)
        
        plt.title("Scatter plot of the estimated outcomes vs the true outcomes")
        plt.xlabel("Estimated Outcome")
        plt.ylabel("True Outcome")
        plt.legend()
        plt.show()
        print('lol')

        # now do the same thing but split up in treatment levels
        for i in range(len(levels)):
            # so need to get the indices of the treatment levels
            indices = np.where(T == i)[0]
            X_embedded_level = X_embedded[indices]
            tarnet_outcomes_pc_level = tarnet_outcomes_pc.loc[case_nr_X[indices]]
            slearner_outcomes_pc_level = slearner_outcomes_pc.loc[case_nr_X[indices]]
            tlearner_outcomes_pc_level = tlearner_outcomes_pc.loc[case_nr_X[indices]]
            ensemble_pc_level = ensemble_pc.loc[case_nr_X[indices]]
            true_outcomes_level = true_outcomes.loc[case_nr_X[indices]]
            # plt.figure(figsize=(8, 6))
            # plt.scatter(X_embedded_level, tarnet_outcomes_pc_level["outcome"], label="TARNET", color='red', alpha=0.5)
            # plt.scatter(X_embedded_level, slearner_outcomes_pc_level["outcome"], label="S-learner", color='#004d00', alpha=0.5)
            # plt.scatter(X_embedded_level, tlearner_outcomes_pc_level["outcome"], label="T-learner", color='purple', alpha=0.5)
            # plt.scatter(X_embedded_level, ensemble_pc_level["outcome"], label="Ensemble", color='blue', alpha=0.5)
            # plt.scatter(X_embedded_level, true_outcomes_level["outcome"], label="True", color='black', alpha=0.5)
            # plt.title(f"TSNE plot of the estimated outcomes vs the true outcomes for treatment level {levels[i]}")
            # plt.xlabel("TSNE")
            # plt.ylabel("Outcome")
            # plt.legend()
            # plt.show()
            # print('lol')

            # just plot the y's for the different treatment levels with on the y-axis the true outcomes and on the x-axis the estimated outcomes
            plt.figure(figsize=(8, 6))
            # plt.scatter(tarnet_outcomes_pc_level["outcome"], true_outcomes_level["outcome"], label="TARNET", color='red', alpha=0.5)
            # plt.scatter(slearner_outcomes_pc_level["outcome"], true_outcomes_level["outcome"], label="S-learner", color='#004d00', alpha=0.5)
            # plt.scatter(tlearner_outcomes_pc_level["outcome"], true_outcomes_level["outcome"], label="T-learner", color='purple', alpha=0.5)
            # plt.scatter(ensemble_pc_level["outcome"], true_outcomes_level["outcome"], label="Ensemble", color='blue', alpha=0.5)

            # Add a line for the true outcomes (y = x line)
            x_values = np.linspace(min(true_outcomes_level["outcome"]), max(true_outcomes_level["outcome"]), 10)
            plt.plot(x_values, x_values, label="True Outcomes", color='black', linestyle='dashed')
            # Add lines for the estimated outcomes
            plt.plot(np.sort(tarnet_outcomes_pc_level["outcome"]), np.sort(true_outcomes_level["outcome"]), color='red', linestyle='solid', alpha=0.3)
            plt.plot(np.sort(slearner_outcomes_pc_level["outcome"]), np.sort(true_outcomes_level["outcome"]), color='#004d00', linestyle='solid', alpha=0.3)
            plt.plot(np.sort(tlearner_outcomes_pc_level["outcome"]), np.sort(true_outcomes_level["outcome"]), color='purple', linestyle='solid', alpha=0.3)
            plt.plot(np.sort(ensemble_pc_level["outcome"]), np.sort(true_outcomes_level["outcome"]), color='blue', linestyle='solid', alpha=0.3)

            plt.title(f"Scatter plot of the estimated outcomes vs the true outcomes for treatment level {levels[i]}")
            plt.xlabel("Estimated Outcome")
            plt.ylabel("True Outcome")
            plt.legend()
            plt.show()
            print('lol')
       


def get_statistical_tests(dataset, intervention, model="LSTM"):
    string_to_add = ""
    if model == "MLP":
        string_to_add = "RealCause"
    uni_metrics_test = load_data(os.path.join(os.getcwd(), dataset, "uni_metrics_test_ensemble_all_" + str(intervention) + string_to_add))
    multi_metrics_test_no_x = load_data(os.path.join(os.getcwd(), dataset, "multi_metrics_test_ensemble_all_no_x_" + str(intervention) + string_to_add))
    multi_metrics_test = load_data(os.path.join(os.getcwd(), dataset, "multi_metrics_test_ensemble_all_" + str(intervention) + string_to_add))
    return uni_metrics_test, multi_metrics_test_no_x, multi_metrics_test

def get_factual_metrics_bpic(dataset, intervention, num_iterations, learner, generator):
    avg_metrics = {}
    se_metrics = {}
    metrics_total = {}
    string_to_add = ""
    if generator == "RealCause":
        string_to_add = "RealCause"
    for iteration in range(num_iterations):
        metrics_total[iteration] = {}
        if learner == "ensemble":
            path_metrics = os.path.join(os.getcwd(), dataset, "metrics_ensemble_" + str(iteration) + "_" + str(intervention) + string_to_add)
        else:
            path_metrics = os.path.join(os.getcwd(), dataset, "metrics_" + str(iteration) + "_" + str(intervention) + str(learner) + string_to_add)
        metrics = load_data(path_metrics)
        if generator == "RealCause" and learner != "ensemble":
            metrics = metrics[0]
        metrics_total[iteration] = metrics
        for key in metrics:
            if key == "n_cases":
                continue
            if key in avg_metrics:
                avg_metrics[key] += metrics[key]
            else:
                avg_metrics[key] = metrics[key]
    for key in avg_metrics:
        avg_metrics[key] = avg_metrics[key] / num_iterations
        se_metrics[key] = np.std([metrics_total[i][key] for i in range(num_iterations)]) / np.sqrt(num_iterations)
    return avg_metrics, se_metrics

def plot_distributions_bpic(intervention, num_iterations):
    for iteration in range(num_iterations):
        # get the merged df of the ensemble
        # merged_df_realcause = load_data(os.path.join(os.getcwd(), "bpic12", "merged_df_" + str(iteration) + "_" + str(intervention) + "TarNet" + "RealCause"))
        
        # Compare in the following way: for t_true = 1 (so factual is outcome1), grab the outcome0 from the procause and realcause, and compare with the value of y_true where t_true = 0
        # factual_df = merged_df_procause[merged_df_procause['t_true'] == 0]
        # df_procause = merged_df_procause[merged_df_procause['t_true'] == 1]
        # df_realcause = merged_df_realcause[merged_df_realcause['t_true'] == 1]

        # # Compute proportions of 0s and 1s for each group
        # counts = pd.DataFrame({
        #     "Method": ["True Outcome"] * 2 + ["ProCause"] * 2 + ["RealCause"] * 2,
        #     "Outcome": [0, 1, 0, 1, 0, 1],
        #     "Proportion": [
        #         (factual_df['y_true'] == 0).mean(), (factual_df['y_true'] == 1).mean(),
        #         (df_procause['outcome0'] == 0).mean(), (df_procause['outcome0'] == 1).mean(),
        #         (df_realcause['outcome0'] == 0).mean(), (df_realcause['outcome0'] == 1).mean()
        #     ]
        # })

        # # Plot as stacked bars for better visualization
        # plt.figure(figsize=(8, 5))
        # sns.barplot(x="Method", y="Proportion", hue="Outcome", data=counts, palette="Set2")

        # # Titles and labels
        # plt.title(f"Distribution of True and Estimated Outcomes (Iteration {iteration})")
        # plt.ylabel("Proportion")
        # plt.ylim(0, 1)
        # plt.legend(title="Outcome", labels=["0", "1"])
        # plt.show()

        # # Now do the same for the t_true = 0, use different colors though
        # factual_df = merged_df_procause[merged_df_procause['t_true'] == 1]
        # df_procause = merged_df_procause[merged_df_procause['t_true'] == 0]
        # df_realcause = merged_df_realcause[merged_df_realcause['t_true'] == 0]

        # # Compute proportions of 0s and 1s for each group
        # counts = pd.DataFrame({
        #     "Method": ["True Outcome"] * 2 + ["ProCause"] * 2 + ["RealCause"] * 2,
        #     "Outcome": [0, 1, 0, 1, 0, 1],
        #     "Proportion": [
        #         (factual_df['y_true'] == 0).mean(), (factual_df['y_true'] == 1).mean(),
        #         (df_procause['outcome1'] == 0).mean(), (df_procause['outcome1'] == 1).mean(),
        #         (df_realcause['outcome1'] == 0).mean(), (df_realcause['outcome1'] == 1).mean()
        #     ]
        # })

        # merged_df_procause_12 = load_data(os.path.join(os.getcwd(), "bpic2012", "merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        # merged_df_procause_17 = load_data(os.path.join(os.getcwd(), "bpic2017", "merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        
        # # Function to compute the mode per group
        # def compute_mode(df, columns_to_mode):
        #     return df.groupby(['case_nr', 'prefix_len'])[columns_to_mode].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).reset_index()

        # # Example usage for both DataFrames
        # columns_to_mode = ['t_est', 'y_est', 't_true', 'y_true']

        # merged_df_procause_12 = compute_mode(merged_df_procause_12, columns_to_mode)
        # merged_df_procause_17 = compute_mode(merged_df_procause_17, columns_to_mode)
        
        # plt.figure(figsize=(12, 10))  # Larger height for 2 rows

        # alpha = 0.2
        # bins = 5
        # font_title = 24

        # # bpic12: Subplot 1 (t_true vs t_est)
        # plt.subplot(2, 2, 1)
        # sns.histplot(merged_df_procause_12['t_true'], color='#1f77b4', label='True', kde=False, bins=bins, stat='probability', alpha=alpha, element='step')
        # sns.histplot(merged_df_procause_12['t_est'], color='#ff7f0e', label='Est', kde=False, bins=bins, stat='probability', alpha=alpha, element='step')
        # # plt.title("BPIC12: Distribution of Treatment Values", fontsize=font_title)
        # plt.title(r"$\mathbf{BPIC12}$: Distribution of Treatment", fontsize=font_title)
        # plt.xticks([0, 1])
        # plt.xlabel("")
        # plt.ylabel("Proportion", fontsize=font_title-2)
        # plt.legend(loc='upper center', fontsize=font_title-2)

        # # bpic12: Subplot 2 (y_true vs y_est)
        # plt.subplot(2, 2, 2)
        # sns.histplot(merged_df_procause_12['y_true'], color='#1f77b4', label='True Y', kde=False, bins=bins, stat='probability', alpha=alpha, element='step')
        # sns.histplot(merged_df_procause_12['y_est'], color='#ff7f0e', label='Est. Y', kde=False, bins=bins, stat='probability', alpha=alpha, element='step')
        # plt.title("Distribution of Outcome", fontsize=font_title)
        # plt.xticks([0, 1])
        # plt.xlabel("")
        # plt.ylabel("")


        # # bpic17: Subplot 3 (t_true vs t_est)
        # plt.subplot(2, 2, 3)
        # sns.histplot(merged_df_procause_17['t_true'], color='#1f77b4', label='True T', kde=False, bins=bins, stat='probability', alpha=alpha, element='step')
        # sns.histplot(merged_df_procause_17['t_est'], color='#ff7f0e', label='Est. T', kde=False, bins=bins, stat='probability', alpha=alpha, element='step')
        # # plt.title("BPIC17: Distribution of Treatment Values", fontsize=font_title)
        # plt.title(r"$\mathbf{BPIC17}$: Distribution of Treatment", fontsize=font_title)
        # plt.xticks([0, 1])
        # plt.xlabel("Treatment Value", fontsize=font_title-2)
        # plt.ylabel("Proportion", fontsize=font_title-2)


        # # bpic17: Subplot 4 (y_true vs y_est)
        # plt.subplot(2, 2, 4)
        # sns.histplot(merged_df_procause_17['y_true'], color='#1f77b4', label='True Y', kde=False, bins=bins, stat='probability', alpha=alpha, element='step')
        # sns.histplot(merged_df_procause_17['y_est'], color='#ff7f0e', label='Est. Y', kde=False, bins=bins, stat='probability', alpha=alpha, element='step')
        # plt.title("Distribution of Outcome", fontsize=font_title)
        # plt.xticks([0, 1])
        # plt.xlabel("Outcome Value", fontsize=font_title-2)
        # plt.ylabel("")

        # plt.tight_layout()

        # # save the figure as pdf
        # plt.savefig(os.path.join(os.getcwd(), "bpic2012", "distribution_plots_" + str(iteration) + "_" + str(intervention) + ".pdf"), bbox_inches='tight')

        # plt.show()





        # Load data
        # check if already processed
        merged_df_procause_12 = load_data(os.path.join(os.getcwd(), "bpic2012", "merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        merged_df_procause_17 = load_data(os.path.join(os.getcwd(), "bpic2017", "merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        # if not os.path.exists(os.path.join(os.getcwd(), "bpic2012", "processed_merged_df_ensemble_" + str(iteration) + "_" + str(intervention))):
        #     merged_df_procause_12 = load_data(os.path.join(os.getcwd(), "bpic2012", "merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        #     merged_df_procause_17 = load_data(os.path.join(os.getcwd(), "bpic2017", "merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))

        #     # Compute mode per group
        #     def compute_mode(df, columns_to_mode):
        #         return df.groupby(['case_nr', 'prefix_len'])[columns_to_mode].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]).reset_index()

        #     columns_to_mode = ['t_est', 'y_est', 't_true', 'y_true']
        #     merged_df_procause_12 = compute_mode(merged_df_procause_12, columns_to_mode)
        #     merged_df_procause_17 = compute_mode(merged_df_procause_17, columns_to_mode)

        #     # save the processed DataFrames
        #     save_data(merged_df_procause_12, os.path.join(os.getcwd(), "bpic2012", "processed_merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        #     save_data(merged_df_procause_17, os.path.join(os.getcwd(), "bpic2017", "processed_merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        # else:
        #     merged_df_procause_12 = load_data(os.path.join(os.getcwd(), "bpic2012", "processed_merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        #     merged_df_procause_17 = load_data(os.path.join(os.getcwd(), "bpic2017", "processed_merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))

        from matplotlib.ticker import MaxNLocator

        def plot_all_datasets(df12, df17):
            prop12 = df12.groupby('prefix_len')[['t_true', 't_est', 'y_true', 'y_est']].mean().reset_index()
            prop17 = df17.groupby('prefix_len')[['t_true', 't_est', 'y_true', 'y_est']].mean().reset_index()

            font_title = 20
            font_label = 20
            font_legend = 18
            lw = 3

            fig, axes = plt.subplots(2, 1, figsize=(7, 8))
            fig.suptitle("Class Distribution per prefix length", fontsize=font_title + 4, y=0.94)

            # BPIC2012 - T & Y
            axes[0].plot(prop12['prefix_len'], prop12['t_true'], label='True T', color='#1f77b4', linestyle='-', linewidth=lw)
            axes[0].plot(prop12['prefix_len'], prop12['t_est'], label='Est. T', color='#ff7f0e', linestyle='--', linewidth=lw)
            axes[0].plot(prop12['prefix_len'], prop12['y_true'], label='True Y', color='#2ca02c', linestyle='-', linewidth=lw)
            axes[0].plot(prop12['prefix_len'], prop12['y_est'], label='Est. Y', color='#d62728', linestyle='--', linewidth=lw)
            axes[0].set_title("BPIC12", fontsize=font_title)
            axes[0].set_ylabel("% Positive (1)", fontsize=font_label)
            axes[0].legend(fontsize=font_legend, loc='center right')
            axes[0].set_xticks(prop12['prefix_len'][::5])
            axes[0].yaxis.set_major_locator(MaxNLocator(nbins=5))

            # BPIC2017 - T & Y
            axes[1].plot(prop17['prefix_len'], prop17['t_true'], label='True T', color='#1f77b4', linestyle='-', linewidth=lw)
            axes[1].plot(prop17['prefix_len'], prop17['t_est'], label='Est. T', color='#ff7f0e', linestyle='--', linewidth=lw)
            axes[1].plot(prop17['prefix_len'], prop17['y_true'], label='True Y', color='#2ca02c', linestyle='-', linewidth=lw)
            axes[1].plot(prop17['prefix_len'], prop17['y_est'], label='Est. Y', color='#d62728', linestyle='--', linewidth=lw)
            axes[1].set_title("BPIC17", fontsize=font_title)
            axes[1].set_xlabel("Prefix Length", fontsize=font_label)
            axes[1].set_ylabel("% Positive (1)", fontsize=font_label)
            # axes[1].legend(fontsize=font_legend)
            axes[1].set_xticks(prop17['prefix_len'][::5])
            axes[1].yaxis.set_major_locator(MaxNLocator(nbins=5))

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            # fig.subplots_adjust(hspace=0.3)

            # Save the figure
            plt.savefig(os.path.join(os.getcwd(), "bpic2012", f"proportion_per_prefix_length_{iteration}_{intervention}.pdf"), bbox_inches='tight')

            plt.show()


        # Call function
        plot_all_datasets(merged_df_procause_12, merged_df_procause_17)


        def plot_bar_proportion_per_prefix(df, dataset_name):
            # Compute mean (proportion of 1s) per prefix_len
            proportions = df.groupby('prefix_len')[['t_true', 't_est', 'y_true', 'y_est']].mean().reset_index()

            font_title = 18
            font_label = 14
            prefix_lens = proportions['prefix_len']
            x = np.arange(len(prefix_lens))
            bar_width = 0.35  # may be increased slightly if needed

            # Select xticks every 5
            tick_interval = 5
            xticks_filtered = [i for i, val in enumerate(prefix_lens) if val % tick_interval == 0]
            xtick_labels_filtered = [prefix_lens[i] for i in xticks_filtered]

            # Calculate dynamic y-axis limits
            t_max = max(proportions['t_true'].max(), proportions['t_est'].max())
            y_max = max(proportions['y_true'].max(), proportions['y_est'].max())
            t_ylim = (0, min(1.0, t_max * 1.2))
            y_ylim = (0, min(1.0, y_max * 1.2))

            fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # Wider figure for spacing
            fig.suptitle(f'{dataset_name} - Proportion of 1s per Prefix Length', fontsize=font_title + 2)

            # Subplot 1: t_true vs t_est
            axes[0].bar(x - bar_width / 2, proportions['t_true'], width=bar_width, label='True T', color='#1f77b4')
            axes[0].bar(x + bar_width / 2, proportions['t_est'], width=bar_width, label='Est. T', color='#ff7f0e')
            axes[0].set_title('Treatment Distribution', fontsize=font_title)
            axes[0].set_xlabel('Prefix Length', fontsize=font_label)
            axes[0].set_ylabel('Proportion of 1s', fontsize=font_label)
            axes[0].set_xticks(xticks_filtered)
            axes[0].set_xticklabels(xtick_labels_filtered, rotation=45)
            axes[0].set_ylim(t_ylim)
            axes[0].legend()
            # axes[0].grid(True, axis='y')

            # Subplot 2: y_true vs y_est
            axes[1].bar(x - bar_width / 2, proportions['y_true'], width=bar_width, label='True Y', color='#1f77b4')
            axes[1].bar(x + bar_width / 2, proportions['y_est'], width=bar_width, label='Est. Y', color='#ff7f0e')
            axes[1].set_title('Outcome Distribution', fontsize=font_title)
            axes[1].set_xlabel('Prefix Length', fontsize=font_label)
            axes[1].set_ylabel('Proportion of 1s', fontsize=font_label)
            axes[1].set_xticks(xticks_filtered)
            axes[1].set_xticklabels(xtick_labels_filtered, rotation=45)
            axes[1].set_ylim(y_ylim)
            axes[1].legend()
            # axes[1].grid(True, axis='y')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


        plot_bar_proportion_per_prefix(merged_df_procause_12, "BPIC2012")
        plot_bar_proportion_per_prefix(merged_df_procause_17, "BPIC2017")



        # # ---- PLOT 3: RealCause - t_true vs t_est ----
        # plt.figure(figsize=(6, 4))
        # sns.histplot(merged_df_realcause['t_est'], color='blue', label='t_est', kde=False, bins=2, stat='probability', alpha=0.6)
        # sns.histplot(merged_df_realcause['t_true'], color='red', label='t_true', kde=False, bins=2, stat='probability', alpha=0.6)
        # plt.title(f"RealCause - t_true vs t_est (Iteration {iteration})")
        # plt.xticks([0, 1])
        # plt.legend()
        # plt.ylabel("Proportion")
        # plt.xlabel("Treatment Value")
        # plt.show()

        # # ---- PLOT 4: RealCause - y_true vs y_est ----
        # plt.figure(figsize=(6, 4))
        # sns.histplot(merged_df_realcause['y_est'], color='blue', label='y_est', kde=False, bins=2, stat='probability', alpha=0.6)
        # sns.histplot(merged_df_realcause['y_true'], color='red', label='y_true', kde=False, bins=2, stat='probability', alpha=0.6)
        # plt.title(f"RealCause - y_true vs y_est (Iteration {iteration})")
        # plt.xticks([0, 1])
        # plt.legend()
        # plt.ylabel("Proportion")
        # plt.xlabel("Outcome Value")
        # plt.show()

def get_ate_bpic(dataset, intervention, num_iterations):
    # calculate the ATE for each iteration
    ate_procause = {}
    ate_realcause = {}

    total_metrics = {"ATE Error ProCause": 0, "ATE Error RealCause": 0, "SE ATE Error ProCause": 0, "SE ATE Error RealCause": 0}

    for iteration in range(num_iterations):
        # get the merged df of the ensemble
        merged_df_procause = load_data(os.path.join(os.getcwd(), dataset, "merged_df_ensemble_" + str(iteration) + "_" + str(intervention)))
        merged_df_realcause = load_data(os.path.join(os.getcwd(), dataset, "merged_df_" + str(iteration) + "_" + str(intervention) + "TarNet" + "RealCause"))

        # get the ATE for the procause and realcause
        ate_procause[iteration] = merged_df_procause['outcome1'].mean() - merged_df_procause['outcome0'].mean()
        ate_realcause[iteration] = merged_df_realcause['outcome1'].mean() - merged_df_realcause['outcome0'].mean()

        # true ATE
        ate_true = merged_df_procause[merged_df_procause["t_true"] == 1]["y_true"].mean() - merged_df_procause[merged_df_procause["t_true"] == 0]["y_true"].mean()
        print("ITERATION", iteration)
        print("True ATE:", ate_true)
        print("ProCause ATE:", ate_procause[iteration])
        print("RealCause ATE:", ate_realcause[iteration])
        print("\n")
    
        error_ate_procause = abs(ate_true - ate_procause[iteration])
        error_ate_realcause = abs(ate_true - ate_realcause[iteration])

        total_metrics["ATE Error ProCause"] += error_ate_procause
        total_metrics["ATE Error RealCause"] += error_ate_realcause
    
    total_metrics["ATE Error ProCause"] = total_metrics["ATE Error ProCause"] / num_iterations
    total_metrics["ATE Error RealCause"] = total_metrics["ATE Error RealCause"] / num_iterations
    total_metrics["SE ATE Error ProCause"] = np.std([abs(ate_true - ate_procause[i]) for i in range(num_iterations)]) / np.sqrt(num_iterations)
    total_metrics["SE ATE Error RealCause"] = np.std([abs(ate_true - ate_realcause[i]) for i in range(num_iterations)]) / np.sqrt(num_iterations)
    
    print('Total metrics:', total_metrics)

    return total_metrics

def fit_knn_bpic(dataset, intervention, n_neighbors_list):
    # get the test set
    data_test_prep = load_data(os.path.join(path, "data", dataset, "data_test_prep_" + str(intervention) + "RealCause"))
    prep_utils = load_data(os.path.join(path, "data", dataset, "prep_utils_" + str(intervention) + "RealCause"))

    #add case_nr and prefix_len to the y_cf_control and y_cf_treated, get it from data_test_prep
    prefix_len_index = prep_utils["column_names"].drop(["outcome", "treatment", "case_nr"]).get_loc("prefix_len")
    prefix_len = data_test_prep['X'][:, prefix_len_index]
    # get unscaled prefix_len, and round to nearest integer
    prefix_len = np.round(prep_utils["scaler_dict_train"]["prefix_len"].inverse_transform(prefix_len.reshape(-1, 1)).flatten())

    y_cf_control_dict = {}
    y_cf_treated_dict = {}

    for n_neighbors in n_neighbors_list:
        knn_evaluator = KNNEvaluator(n_neighbors=n_neighbors)
        # get the X and y of the control group, by getting which indices in 'T' have as value 0 (please note T, X and Y are tensors)
        control_indices = np.where(data_test_prep['T'] == 0)[0]
        X_control = data_test_prep['X'][control_indices]
        y_control = data_test_prep['Y'][control_indices]

        treated_indices = np.where(data_test_prep['T'] == 1)[0]
        X_treated = data_test_prep['X'][treated_indices]
        y_treated = data_test_prep['Y'][treated_indices]

        # fit the knn model for each group
        knn_evaluator.fit(X_control=X_control, y_control=y_control, X_treated=X_treated, y_treated=y_treated)

        # now predict the outcomes for the control group with the treated group knn model
        y_cf_control = knn_evaluator.predict(X=X_control, predict_cf_control=True)
        y_cf_treated = knn_evaluator.predict(X=X_treated, predict_cf_control=False)

        y_cf_control = pd.DataFrame(y_cf_control, columns=['outcome1'])
        y_cf_control['case_nr'] = data_test_prep['case_nr'][control_indices]
        y_cf_control['prefix_len'] = prefix_len[control_indices]
        y_cf_treated = pd.DataFrame(y_cf_treated, columns=['outcome0'])
        y_cf_treated['case_nr'] = data_test_prep['case_nr'][treated_indices]
        y_cf_treated['prefix_len'] = prefix_len[treated_indices]

        y_cf_control_dict[n_neighbors] = y_cf_control
        y_cf_treated_dict[n_neighbors] = y_cf_treated

    return y_cf_control_dict, y_cf_treated_dict

# def get_knn_metrics_bpic(dataset, intervention, num_iterations, y_cf_control, y_cf_treated):
#     cf_of_control_avg = {"ProCause": {"F1": 0, "ROC_AUC": 0}, "RealCause": {"F1": 0, "ROC_AUC": 0}}
#     cf_of_treated_avg = {"ProCause": {"F1": 0, "ROC_AUC": 0}, "RealCause": {"F1": 0, "ROC_AUC": 0}}
#     cf_of_control_se = {"ProCause": {"F1": [], "ROC_AUC": []}, "RealCause": {"F1": [], "ROC_AUC": []}}
#     cf_of_treated_se = {"ProCause": {"F1": [], "ROC_AUC": []}, "RealCause": {"F1": [], "ROC_AUC": []}}

#     cf_total_avg = {"ProCause": {"F1": 0, "ROC_AUC": 0}, "RealCause": {"F1": 0, "ROC_AUC": 0}}
#     cf_total_se = {"ProCause": {"F1": [], "ROC_AUC": []}, "RealCause": {"F1": [], "ROC_AUC": []}}
    
#     for iteration in range(num_iterations):
#         merged_df_procause = load_data(os.path.join(os.getcwd(), dataset, f"merged_df_ensemble_{iteration}_{intervention}"))
#         merged_df_realcause = load_data(os.path.join(os.getcwd(), dataset, f"merged_df_{iteration}_{intervention}TarNetRealCause"))

#         # get the counterfactual outcome for the control and treated group, and grab the mode for each case_nr and prefix_len combination (so just grab the mean and round it)
#         y_cf_control_pred_procause = merged_df_procause[merged_df_procause["t_true"] == 0]["outcome1"].groupby([merged_df_procause["case_nr"], merged_df_procause["prefix_len"]]).mean().round()
#         y_cf_treated_pred_procause = merged_df_procause[merged_df_procause["t_true"] == 1]["outcome0"].groupby([merged_df_procause["case_nr"], merged_df_procause["prefix_len"]]).mean().round()
#         y_cf_control_pred_realcause = merged_df_realcause[merged_df_realcause["t_true"] == 0]["outcome1"].groupby([merged_df_realcause["case_nr"], merged_df_realcause["prefix_len"]]).mean().round()
#         y_cf_treated_pred_realcause = merged_df_realcause[merged_df_realcause["t_true"] == 1]["outcome0"].groupby([merged_df_realcause["case_nr"], merged_df_realcause["prefix_len"]]).mean().round()

#         f1_procause_cf_of_control = f1_score(y_cf_control, y_cf_control_pred_procause)
#         f1_procause_cf_of_treated = f1_score(y_cf_treated, y_cf_treated_pred_procause)
#         f1_realcause_cf_of_control = f1_score(y_cf_control, y_cf_control_pred_realcause)
#         f1_realcause_cf_of_treated = f1_score(y_cf_treated, y_cf_treated_pred_realcause)
        
#         roc_auc_procause_cf_of_control = roc_auc_score(y_cf_control, y_cf_control_pred_procause)
#         roc_auc_procause_cf_of_treated = roc_auc_score(y_cf_treated, y_cf_treated_pred_procause)
#         roc_auc_realcause_cf_of_control = roc_auc_score(y_cf_control, y_cf_control_pred_realcause)
#         roc_auc_realcause_cf_of_treated = roc_auc_score(y_cf_treated, y_cf_treated_pred_realcause)
        
#         for key, f1_val, roc_val in zip(
#             ["ProCause", "RealCause"],
#             [[f1_procause_cf_of_control, f1_procause_cf_of_treated], [f1_realcause_cf_of_control, f1_realcause_cf_of_treated]],
#             [[roc_auc_procause_cf_of_control, roc_auc_procause_cf_of_treated], [roc_auc_realcause_cf_of_control, roc_auc_realcause_cf_of_treated]]
#         ):
#             cf_of_control_avg[key]["F1"] += f1_val[0]
#             cf_of_treated_avg[key]["F1"] += f1_val[1]
#             cf_of_control_avg[key]["ROC_AUC"] += roc_val[0]
#             cf_of_treated_avg[key]["ROC_AUC"] += roc_val[1]
            
#             cf_of_control_se[key]["F1"].append(f1_val[0])
#             cf_of_treated_se[key]["F1"].append(f1_val[1])
#             cf_of_control_se[key]["ROC_AUC"].append(roc_val[0])
#             cf_of_treated_se[key]["ROC_AUC"].append(roc_val[1])

#     for key in ["ProCause", "RealCause"]:
#         for metric in ["F1", "ROC_AUC"]:
#             cf_of_control_avg[key][metric] /= num_iterations
#             cf_of_treated_avg[key][metric] /= num_iterations
#             cf_of_control_se[key][metric] = np.std(cf_of_control_se[key][metric]) / np.sqrt(num_iterations)
#             cf_of_treated_se[key][metric] = np.std(cf_of_treated_se[key][metric]) / np.sqrt(num_iterations)

#             cf_total_avg[key][metric] = (cf_of_control_avg[key][metric] + cf_of_treated_avg[key][metric]) / 2
#             cf_total_se[key][metric] = np.sqrt(cf_of_control_se[key][metric] ** 2 + cf_of_treated_se[key][metric] ** 2) / 2
    
#     metrics = {
#         f"CF of Control {key} {metric}": cf_of_control_avg[key][metric]
#         for key in cf_of_control_avg for metric in cf_of_control_avg[key]
#     }
#     metrics.update({
#         f"CF of Treated {key} {metric}": cf_of_treated_avg[key][metric]
#         for key in cf_of_treated_avg for metric in cf_of_treated_avg[key]
#     })
#     metrics.update({
#         f"SE CF of Control {key} {metric}": cf_of_control_se[key][metric]
#         for key in cf_of_control_se for metric in cf_of_control_se[key]
#     })
#     metrics.update({
#         f"SE CF of Treated {key} {metric}": cf_of_treated_se[key][metric]
#         for key in cf_of_treated_se for metric in cf_of_treated_se[key]
#     })

#     metrics.update({
#         f"Total CF {key} {metric}": cf_total_avg[key][metric]
#         for key in cf_total_avg for metric in cf_total_avg[key]
#     })

#     metrics.update({
#         f"SE Total CF {key} {metric}": cf_total_se[key][metric]
#         for key in cf_total_se for metric in cf_total_se[key]
#     })

#     for key, val in metrics.items():
#         print(f"{key}: {val}")
#     print("\n")
    
#     return metrics

def get_knn_metrics_bpic(dataset, intervention, num_iterations, y_cf_control, y_cf_treated):
    cf_avg = {"ProCause": {"F1": 0, "ROC_AUC": 0}, "RealCause": {"F1": 0, "ROC_AUC": 0}}
    cf_se = {"ProCause": {"F1": [], "ROC_AUC": []}, "RealCause": {"F1": [], "ROC_AUC": []}}
    
    cf_control_avg = {"ProCause": {"F1": 0, "ROC_AUC": 0}, "RealCause": {"F1": 0, "ROC_AUC": 0}}
    cf_treated_avg = {"ProCause": {"F1": 0, "ROC_AUC": 0}, "RealCause": {"F1": 0, "ROC_AUC": 0}}
    cf_control_se = {"ProCause": {"F1": [], "ROC_AUC": []}, "RealCause": {"F1": [], "ROC_AUC": []}}
    cf_treated_se = {"ProCause": {"F1": [], "ROC_AUC": []}, "RealCause": {"F1": [], "ROC_AUC": []}}
    
    for iteration in range(num_iterations):
        merged_df_procause = load_data(os.path.join(os.getcwd(), dataset, f"merged_df_ensemble_{iteration}_{intervention}"))
        merged_df_realcause = load_data(os.path.join(os.getcwd(), dataset, f"merged_df_{iteration}_{intervention}TarNetRealCause"))

        def compute_weighted_metrics(merged_df, y_cf_control, y_cf_treated):
            prefix_counts = merged_df.groupby("prefix_len")["case_nr"].nunique()
            total_cases = prefix_counts.sum()
            weights = (prefix_counts / total_cases).to_dict()
            
            weighted_f1, weighted_roc_auc = {"F1": 0, "ROC_AUC": 0}, {"F1": 0, "ROC_AUC": 0}
            weighted_f1_control, weighted_f1_treated = 0, 0
            weighted_roc_auc_control, weighted_roc_auc_treated = 0, 0
            
            for prefix_len, weight in weights.items():
                knn_control = y_cf_control[y_cf_control["prefix_len"] == prefix_len]
                pred_control = merged_df[(merged_df["t_true"] == 0) & (merged_df["prefix_len"] == prefix_len)]

                knn_treated = y_cf_treated[y_cf_treated["prefix_len"] == prefix_len]
                pred_treated = merged_df[(merged_df["t_true"] == 1) & (merged_df["prefix_len"] == prefix_len)]

                pred_control_mode = pred_control.groupby("case_nr", as_index=False).mean().round()
                pred_control_mode_matched = knn_control[["case_nr"]].merge(pred_control_mode, on="case_nr", how="left")

                pred_treated_mode = pred_treated.groupby("case_nr", as_index=False).mean().round()
                pred_treated_mode_matched = knn_treated[["case_nr"]].merge(pred_treated_mode, on="case_nr", how="left")

                y_cf_control_current = y_cf_control[y_cf_control["prefix_len"] == prefix_len]["outcome1"]
                y_cf_treated_current = y_cf_treated[y_cf_treated["prefix_len"] == prefix_len]["outcome0"]

                y_cf_control_pred_current = pred_control_mode_matched["outcome1"]
                y_cf_treated_pred_current = pred_treated_mode_matched["outcome0"]

                if len(y_cf_control_pred_current) > 0:
                    f1_control = f1_score(y_cf_control_current, y_cf_control_pred_current)
                    # check if more than 1 class is present
                    if len(np.unique(y_cf_control_current)) > 1:
                        roc_auc_control = roc_auc_score(y_cf_control_current, y_cf_control_pred_current)
                    else:
                        roc_auc_control = 0.5
                    weighted_f1_control += weight * f1_control
                    weighted_roc_auc_control += weight * roc_auc_control
                
                if len(y_cf_treated_pred_current) > 0:
                    f1_treated = f1_score(y_cf_treated_current, y_cf_treated_pred_current)
                    # check if more than 1 class is present
                    if len(np.unique(y_cf_treated_current)) > 1:
                        roc_auc_treated = roc_auc_score(y_cf_treated_current, y_cf_treated_pred_current)
                    else:
                        roc_auc_treated = 0.5
                    weighted_f1_treated += weight * f1_treated
                    weighted_roc_auc_treated += weight * roc_auc_treated
                
            weighted_f1["F1"] = (weighted_f1_control + weighted_f1_treated) / 2
            weighted_roc_auc["ROC_AUC"] = (weighted_roc_auc_control + weighted_roc_auc_treated) / 2
            
            return weighted_f1, weighted_roc_auc, weighted_f1_control, weighted_f1_treated, weighted_roc_auc_control, weighted_roc_auc_treated
        
        for key, merged_df in zip(["ProCause", "RealCause"], [merged_df_procause, merged_df_realcause]):
            weighted_f1, weighted_roc_auc, weighted_f1_control, weighted_f1_treated, weighted_roc_auc_control, weighted_roc_auc_treated = compute_weighted_metrics(merged_df, y_cf_control, y_cf_treated)
            
            cf_avg[key]["F1"] += weighted_f1["F1"]
            cf_avg[key]["ROC_AUC"] += weighted_roc_auc["ROC_AUC"]
            cf_se[key]["F1"].append(weighted_f1["F1"])
            cf_se[key]["ROC_AUC"].append(weighted_roc_auc["ROC_AUC"])
            
            cf_control_avg[key]["F1"] += weighted_f1_control
            cf_control_avg[key]["ROC_AUC"] += weighted_roc_auc_control
            cf_control_se[key]["F1"].append(weighted_f1_control)
            cf_control_se[key]["ROC_AUC"].append(weighted_roc_auc_control)
            
            cf_treated_avg[key]["F1"] += weighted_f1_treated
            cf_treated_avg[key]["ROC_AUC"] += weighted_roc_auc_treated
            cf_treated_se[key]["F1"].append(weighted_f1_treated)
            cf_treated_se[key]["ROC_AUC"].append(weighted_roc_auc_treated)
    
    for key in ["ProCause", "RealCause"]:
        for metric in ["F1", "ROC_AUC"]:
            cf_avg[key][metric] /= num_iterations
            cf_se[key][metric] = np.std(cf_se[key][metric]) / np.sqrt(num_iterations)
            
            cf_control_avg[key][metric] /= num_iterations
            cf_control_se[key][metric] = np.std(cf_control_se[key][metric]) / np.sqrt(num_iterations)
            
            cf_treated_avg[key][metric] /= num_iterations
            cf_treated_se[key][metric] = np.std(cf_treated_se[key][metric]) / np.sqrt(num_iterations)
    
    metrics = {f"Weighted CF {key} {metric}": cf_avg[key][metric] for key in cf_avg for metric in cf_avg[key]}
    metrics.update({f"SE Weighted CF {key} {metric}": cf_se[key][metric] for key in cf_se for metric in cf_se[key]})
    
    metrics.update({f"Weighted CF Control {key} {metric}": cf_control_avg[key][metric] for key in cf_control_avg for metric in cf_control_avg[key]})
    metrics.update({f"SE Weighted CF Control {key} {metric}": cf_control_se[key][metric] for key in cf_control_se for metric in cf_control_se[key]})
    
    metrics.update({f"Weighted CF Treated {key} {metric}": cf_treated_avg[key][metric] for key in cf_treated_avg for metric in cf_treated_avg[key]})
    metrics.update({f"SE Weighted CF Treated {key} {metric}": cf_treated_se[key][metric] for key in cf_treated_se for metric in cf_treated_se[key]})
    
    for key, val in metrics.items():
        print(f"{key}: {val}")
    print("\n")
    
    return metrics
