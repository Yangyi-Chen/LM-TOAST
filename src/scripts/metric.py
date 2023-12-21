import numpy as np
import math
import os
import argparse
from sklearn import metrics
# from statsmodels.stats.weightstats import ztest
# from scipy.stats import ttest_1samp, levene, ttest_ind


def compute_ece(allprobs_list, allpreds_list, alllabels_list):
    if not isinstance(allprobs_list[0], list):
        allprobs_list = [allprobs_list]
        allpreds_list = [allpreds_list]
        alllabels_list = [alllabels_list]

    acc_list = []
    avg_prob_list = []
    ECE_equal_interval_list = []
    ECE_equal_interval_subset_list = {0: [], 1: []}
    ECE_equal_mass_list = []
    ECE_equal_mass_subset_list = {0: [], 1: []}
    pvalue_on_true_list = []
    pvalue_on_false_list = []
    pvalue_list = []
    division_list = []
    pvalue_z_on_true_list = []
    pvalue_z_on_false_list = []
    # pvalue_z_list = []
    auroc_list = []
    for allprobs, allpreds, alllabels in zip(allprobs_list, allpreds_list, alllabels_list):

        pred = [1-item for item in allprobs]
        y = [allpreds[k] != alllabels[k] for k in range(len(allpreds))]

        output = metrics.roc_curve(y, pred)  # fpr, tpr, threshold
        auc = metrics.auc(output[0], output[1])
        auroc_list.append(auc)



        avg_prob = np.mean(allprobs)
        avg_prob_list.append(avg_prob)
        acc = [int(i == j) for i, j in zip(allpreds, alllabels)]

        # ECE_equal_interval_subset = calibrate_on_subset_equal_interval(acc, allprobs, allpreds, alllabels, flags=None)
        # ECE_equal_interval_subset_list[0].append(ECE_equal_interval_subset[0])
        # ECE_equal_interval_subset_list[1].append(ECE_equal_interval_subset[1])
        ECE_equal_interval_subset_list[0].append(
            np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred != label]))
        ECE_equal_interval_subset_list[1].append(
            1 - np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred == label]))

        # ECE_equal_mass_subset = calibrate_on_subset_equal_mass(acc, allprobs, allpreds, alllabels, flags=None)
        # ECE_equal_mass_subset_list[0].append(ECE_equal_mass_subset[0])
        # ECE_equal_mass_subset_list[1].append(ECE_equal_mass_subset[1])
        neg_val = np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred != label])
        pos_val = np.mean([prob for (prob, pred, label) in zip(allprobs, allpreds, alllabels) if pred == label])
        ECE_equal_mass_subset_list[0].append(neg_val)
        ECE_equal_mass_subset_list[1].append(1-pos_val)

        division_list.append(pos_val-neg_val)



        acc = sum(acc) / len(acc)
        acc_list.append(acc)

        ECE_equal_interval = calibrate_on_subset_equal_interval([0] * len(alllabels), allprobs, allpreds, alllabels,
                                                                flags=None)
        ECE_equal_interval_list.append(ECE_equal_interval[0])

        ECE_equal_mass = calibrate_on_subset_equal_mass([0] * len(alllabels), allprobs, allpreds, alllabels, flags=None)
        ECE_equal_mass_list.append(ECE_equal_mass[0])

        # # t-test
        # allprobs_on_true = [prob for prob, pred, label in zip(allprobs, allpreds, alllabels) if pred==label]
        # allprobs_on_false = [prob for prob, pred, label in zip(allprobs, allpreds, alllabels) if pred!=label]
        # tstat_z_on_true, pvalue_z_on_true = ztest(allprobs_on_true)
        # tstat_z_on_false, pvalue_z_on_false = ztest(allprobs_on_false)
        # pvalue_z_on_true_list.append(pvalue_z_on_true)
        # pvalue_z_on_false_list.append(pvalue_z_on_false)
        # # pvalue_z_list.append()

        # tstat_on_true, pvalue_on_true = ttest_1samp(allprobs_on_true, 1)
        # tstat_on_false, pvalue_on_false = ttest_1samp(allprobs_on_false, 1/len(set(alllabels)))
        # pvalue_on_true_list.append(pvalue_on_true)
        # pvalue_on_false_list.append(pvalue_on_false)

        # # 进行levene检验（检验方差齐性）
        # _, levene_p = levene(allprobs_on_true, allprobs_on_false)
        # equal_var = True if levene_p > 0.05 else False
        # tstat, pvalue = ttest_ind(allprobs_on_true, allprobs_on_false, equal_var=equal_var)
        # pvalue_list.append(pvalue)

    avg_ECE_equal_interval_subset = {0: np.mean(ECE_equal_interval_subset_list[0]),
                                     1: np.mean(ECE_equal_interval_subset_list[1])}
    std_ECE_equal_interval_subset = {0: np.std([item*100 for item in ECE_equal_interval_subset_list[0]]),
                                     1: np.std([item * 100 for item in ECE_equal_interval_subset_list[1]])}
    avg_ECE_equal_interval = np.mean(ECE_equal_interval_list)
    std_ECE_equal_interval = np.std([item*100 for item in ECE_equal_interval_list])
    avg_ECE_equal_mass_subset = {0: np.mean(ECE_equal_mass_subset_list[0]), 1: np.mean(ECE_equal_mass_subset_list[1])}
    std_ECE_equal_mass_subset = {0: np.std([item*100 for item in ECE_equal_mass_subset_list[0]]), 1: np.std([item*100 for item in ECE_equal_mass_subset_list[1]])}
    avg_ECE_equal_mass = np.mean(ECE_equal_mass_list)
    std_ECE_equal_mass = np.std([item*100 for item in ECE_equal_mass_list])

    # avg_pvalue_on_true = np.mean(pvalue_on_true_list)
    # avg_pvalue_on_false = np.mean(pvalue_on_false_list)
    # avg_pvalue = np.mean(pvalue_list)

    # avg_pvalue_z_on_true = np.mean(pvalue_z_on_true_list)

    # avg_pvalue_z_on_false = np.mean(pvalue_z_on_false_list)
    # avg_pvalue_z = np.mean(pvalue_z_list)

    avg_acc = np.mean(acc_list)
    std_acc = np.std(acc_list)
    avg_probs = np.mean(avg_prob_list)
    std_probs = np.std([item*100 for item in avg_prob_list])

    print("acc:", avg_acc, std_acc)
    print("avg_probs:", avg_probs, std_probs)
    print("|avg_acc-avg_prob| =", abs(avg_acc - avg_probs))

    for key in range(2):
        print(f"ECE_equal_interval on subsets [{key}]:", avg_ECE_equal_interval_subset[key],
              std_ECE_equal_interval_subset[key])

    print("ECE_equal_interval: ", avg_ECE_equal_interval, std_ECE_equal_interval)

    for key in range(2):
        print(f"ECE_equal_mass on subsets [{key}]:", avg_ECE_equal_mass_subset[key], std_ECE_equal_mass_subset[key])

    print("ECE_equal_mass: ", avg_ECE_equal_mass, std_ECE_equal_mass)


    print("Conf Div: {}, std: {}".format(np.average(division_list), np.std([item*100 for item in division_list])))
    print("AUROC: {}, std: {}".format(np.average(auroc_list), np.std([item*100 for item in auroc_list])))







    # print("pvalue_on_true:", avg_pvalue_on_true)
    # print("pvalue_on_false:", avg_pvalue_on_false)
    # print("pvalue:", avg_pvalue)
    print()

    with open(f"./results/metrics/{setting}/{dataset_name}/{model_name}.tsv", "a") as f:
        print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(avg_acc, avg_probs,
                                                                                              abs(avg_acc - avg_probs),
                                                                                              avg_ECE_equal_interval,
                                                                                              avg_ECE_equal_interval_subset[
                                                                                                  1],
                                                                                              avg_ECE_equal_interval_subset[
                                                                                                  0],
                                                                                              avg_ECE_equal_mass,
                                                                                              avg_ECE_equal_mass_subset[
                                                                                                  1],
                                                                                              avg_ECE_equal_mass_subset[
                                                                                                  0],
                                                                                              std_acc, std_probs,
                                                                                              abs(std_acc - std_probs),
                                                                                              std_ECE_equal_interval,
                                                                                              std_ECE_equal_interval_subset[
                                                                                                  1],
                                                                                              std_ECE_equal_interval_subset[
                                                                                                  0],
                                                                                              std_ECE_equal_mass,
                                                                                              std_ECE_equal_mass_subset[
                                                                                                  1],
                                                                                              std_ECE_equal_mass_subset[
                                                                                                  0]), file=f)


def calibrate_on_common_subset(acc_pretrain, allprobs_pretrain, allpreds_pretrain, alllabels_pretrain,
                               acc_random, allprobs_random, allpreds_random, alllabels_random):
    flags = [int(i == j) for i, j in zip(acc_pretrain, acc_random)]

    ECE_equal_interval_pretrain = calibrate_on_subset_equal_interval(acc_pretrain,
                                                                     allprobs_pretrain, allpreds_pretrain,
                                                                     alllabels_pretrain, flags)
    ECE_equal_interval_random = calibrate_on_subset_equal_interval(acc_random,
                                                                   allprobs_random, allpreds_random, alllabels_random,
                                                                   flags)

    ECE_equal_mass_pretrain = calibrate_on_subset_equal_mass(acc_pretrain,
                                                             allprobs_pretrain, allpreds_pretrain, alllabels_pretrain,
                                                             flags)
    ECE_equal_mass_random = calibrate_on_subset_equal_mass(acc_random,
                                                           allprobs_random, allpreds_random, alllabels_random, flags)

    # print("on commen subsets:")
    # for key in [0, 1]:
    # print(f"ECE_euqal_interval_pretrain[{key}]:", ECE_equal_interval_pretrain[key])
    # print(f"ECE_euqal_interval_random[{key}]:", ECE_equal_interval_random[key])
    # for key in [0, 1]:
    # print(f"ECE_euqal_mass_pretrain[{key}]:", ECE_equal_mass_pretrain[key])
    # print(f"ECE_euqal_mass_random[{key}]:", ECE_equal_mass_random[key])
    # print()


def calibrate_on_subset_equal_interval(accuracies, allprobs, allpreds, alllabels, flags=None):
    if flags is None:
        flags = [1] * len(accuracies)
    probs_of_bins = {0: {}, 1: {}}  # 0: false, 1: true
    preds_of_bins = {0: {}, 1: {}}
    labels_of_bins = {0: {}, 1: {}}
    labels_of_subsets = {0: [], 1: []}

    for key in [0, 1]:
        labels_of_subsets[key] = []
        for bin in range(1, 11):
            probs_of_bins[key][bin] = []
            preds_of_bins[key][bin] = []
            labels_of_bins[key][bin] = []

    for flag, key, prob, pred, label in zip(flags, accuracies, allprobs, allpreds, alllabels):
        if flag == 1:
            bin = math.ceil(prob * 10)
            probs_of_bins[key][bin].append(prob)
            preds_of_bins[key][bin].append(pred)
            labels_of_bins[key][bin].append(label)
            labels_of_subsets[key].append(label)

    ECE = {}
    for key in [0, 1]:
        ECE_of_subsets = 0
        for bin in range(1, 11):
            probs = probs_of_bins[key][bin]
            preds = preds_of_bins[key][bin]
            labels = labels_of_bins[key][bin]
            avg_probs = sum([prob for prob in probs]) / len(probs) if len(probs) != 0 else 0
            bin_acc = sum([int(i == j) for i, j in zip(preds, labels)]) / len(probs) if len(probs) != 0 else 0
            # #print("bin: {}, bin_acc: {}, avg_prob: {}".format(bin, bin_acc, avg_probs))
            ECE_of_subsets += abs(bin_acc - avg_probs) * len(probs)

        if ECE_of_subsets != 0 and len(labels_of_subsets[key]) == 0:
            # print("error")
            exit()
        ECE[key] = ECE_of_subsets / len(labels_of_subsets[key]) if len(labels_of_subsets[key]) != 0 else 0
        # #print("ECE[key] = ECE_of_subsets / len(labels_of_subsets[acc]) = {} / {} = {}".format(ECE_of_subsets, len(labels_of_subsets[key]), ECE[key]))

    return ECE


def calibrate_on_subset_equal_mass(accuracies, allprobs, allpreds, alllabels, flags=None):
    if flags is None:
        flags = [1] * len(accuracies)
    probs_of_bins = {0: {}, 1: {}}  # 0: false, 1: true
    preds_of_bins = {0: {}, 1: {}}
    labels_of_bins = {0: {}, 1: {}}
    labels_of_subsets = {0: [], 1: []}

    for key in [0, 1]:
        labels_of_subsets[key] = []
        for bin in range(100):
            probs_of_bins[key][bin] = []
            preds_of_bins[key][bin] = []
            labels_of_bins[key][bin] = []

    # sort by prob
    data = zip(allprobs, allpreds, alllabels, flags, accuracies)
    data = sorted(data)
    allprobs, allpreds, alllabels, flags, accuracies = zip(*data)
    bin_num = 100
    num_samples_per_bin = math.ceil(len(flags) / bin_num)
    for i, (flag, key, prob, pred, label) in enumerate(zip(flags, accuracies, allprobs, allpreds, alllabels)):
        if flag == 1:
            bin = int(i / num_samples_per_bin)
            probs_of_bins[key][bin].append(prob)
            preds_of_bins[key][bin].append(pred)
            labels_of_bins[key][bin].append(label)
            labels_of_subsets[key].append(label)

    ECE = {}
    for key in [0, 1]:
        ECE_of_subsets = 0
        for bin in range(100):
            probs = probs_of_bins[key][bin]
            preds = preds_of_bins[key][bin]
            labels = labels_of_bins[key][bin]
            avg_probs = sum([prob for prob in probs]) / len(probs) if len(probs) != 0 else 0
            bin_acc = sum([int(i == j) for i, j in zip(preds, labels)]) / len(probs) if len(probs) != 0 else 0
            # #print("bin: {}, bin_acc: {}, avg_prob: {}".format(bin, bin_acc, avg_probs))
            ECE_of_subsets += abs(bin_acc - avg_probs) * len(probs)

        if ECE_of_subsets != 0 and len(labels_of_subsets[key]) == 0:
            # print("error")
            exit()
        ECE[key] = ECE_of_subsets / len(labels_of_subsets[key]) if len(labels_of_subsets[key]) != 0 else 0
        # #print("ECE[key] = ECE_of_subsets / len(labels_of_subsets[acc]) = {} / {} = {}".format(ECE_of_subsets, len(labels_of_subsets[key]), ECE[key]))

    return ECE


def pretrain():
    for model in [f"{model_name}-pretrain", f"{model_name}-random", "lstm", "tf_idf", "bag_of_words"]:
        allprobs = np.load(f"./results/pretrain/{dataset_name}/{model}/allprobs.npy").tolist()
        allpreds = np.load(f"./results/pretrain/{dataset_name}/{model}/allpreds.npy").tolist()
        alllabels = np.load(f"./results/pretrain/{dataset_name}/{model}/alllabels.npy").tolist()

        compute_ece(allprobs, allpreds, alllabels)


def kshots():
    for shot in [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192, 9216,
                 10240, 11264, 12288, 13312, 16384, 32768, 65536, 131072, 262144]:
        if not os.path.exists(f"./results/kshots/{dataset_name}/{model_name}/{shot}-shots"):
            continue
        # print(f"on {shot} shots:")
        allprobs_list = []
        allpreds_list = []
        alllabels_list = []
        for seed in range(5):
            if not os.path.exists(f"./results/kshots/{dataset_name}/{model_name}/{shot}-shots/{seed}"):
                continue
            allprobs = np.load(
                f"./results/kshots/{dataset_name}/{model_name}/{shot}-shots/{seed}/allprobs.npy").tolist()
            allpreds = np.load(
                f"./results/kshots/{dataset_name}/{model_name}/{shot}-shots/{seed}/allpreds.npy").tolist()
            alllabels = np.load(
                f"./results/kshots/{dataset_name}/{model_name}/{shot}-shots/{seed}/alllabels.npy").tolist()
            # print(sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(alllabels))
            allprobs_list.append(allprobs)
            allpreds_list.append(allpreds)
            alllabels_list.append(alllabels)

        compute_ece(allprobs_list, allpreds_list, alllabels_list)


def single_model():
    alllabels = np.load(f"./results/pretrain/{dataset_name}/{model_name}/alllabels.npy")
    allprobs = np.load(f"./results/pretrain/{dataset_name}/{model_name}/allprobs.npy")
    allpreds = np.load(f"./results/pretrain/{dataset_name}/{model_name}/allpreds.npy")
    compute_ece(allprobs, allpreds, alllabels)


def dynamics():
    STEP = {
        "sst2": 100,
        "agnews": 100,
        "mnli": 5000,
        "yahoo_answers_topics": 2000
    }
    step = STEP[dataset_name]
    for step in range(0, 100000000, step):
        if not os.path.exists(f"./results/dynamics/{dataset_name}/{model_name}/{step}"):
            break
        # print(f"at step {step}:")
        alllabels = np.load(f"./results/dynamics/{dataset_name}/{model_name}/{step}/alllabels.npy")
        allprobs = np.load(f"./results/dynamics/{dataset_name}/{model_name}/{step}/allprobs.npy")
        allpreds = np.load(f"./results/dynamics/{dataset_name}/{model_name}/{step}/allpreds.npy")
        compute_ece(allprobs, allpreds, alllabels)


def scale():
    sizes = ["tiny", "mini", "small", "medium", "base", "large"] if model_name == "bert" \
        else ["small", "base", "large", "3b"]
    for size in sizes:
        if not os.path.exists(f"./results/scale/{dataset_name}/{model_name}/{size}"):
            break
        # print(f"at size {size}:")
        alllabels = np.load(f"./results/scale/{dataset_name}/{model_name}/{size}/alllabels.npy")
        allprobs = np.load(f"./results/scale/{dataset_name}/{model_name}/{size}/allprobs.npy")
        allpreds = np.load(f"./results/scale/{dataset_name}/{model_name}/{size}/allpreds.npy")
        compute_ece(allprobs, allpreds, alllabels)


def parameter():
    seed = 0
    for bottleneck_dim in [1, 4, 16, 64, 256, 1024]:
        if not os.path.exists(f"./results/parameter/{dataset_name}/{model_name}/{bottleneck_dim}-dim/{seed}"):
            continue
        # print(f"at {bottleneck_dim} dim:")
        alllabels = np.load(
            f"./results/parameter/{dataset_name}/{model_name}/{bottleneck_dim}-dim/{seed}/alllabels.npy")
        allprobs = np.load(f"./results/parameter/{dataset_name}/{model_name}/{bottleneck_dim}-dim/{seed}/allprobs.npy")
        allpreds = np.load(f"./results/parameter/{dataset_name}/{model_name}/{bottleneck_dim}-dim/{seed}/allpreds.npy")
        compute_ece(allprobs, allpreds, alllabels)


def soft_prompt():
    seed = 0
    for soft_token_num in [1, 5, 10, 20, 50]:

        if not os.path.exists(
                f"./results/parameter/soft_prompt/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}"):
            print(f"./results/parameter/soft_prompt/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}")
            continue
        # print(f"at {bottleneck_dim} dim:")
        alllabels = np.load(
            f"./results/parameter/soft_prompt/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}/alllabels.npy")
        allprobs = np.load(
            f"./results/parameter/soft_prompt/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}/allprobs.npy")
        allpreds = np.load(
            f"./results/parameter/soft_prompt/{dataset_name}/{model_name}/{soft_token_num}-token/{seed}/allpreds.npy")
        compute_ece(allprobs, allpreds, alllabels)


def ood():
    OOD_DATASET = {
        "mnli": ["mnli_iid", "hans", "anli"],
        "amazon_food": ["amazon_food_iid", "sst5", "semeval"],
        "civil_comments": ["civil_comments_iid", "hate_speech", "implicit_hate"],
        "dynasent": ["dynasent_iid", "amazon_food", "dsc"]
    }

    if model_name in ["t5-base-small", "t5-base-middle", "t5-small-middle", "t5-large-middle"]:
        method_list = ["feature-based", "verbalized", "verbalized-self", "verbalized-iterative", "verbalized-multitask"]
    else:
        method_list = ["None", "temperature_scaling", "label_smoothing", "eda", "ensemble",
                       "feature-based", "verbalized", "verbalized-self", "verbalized-iterative", "verbalized-multitask"]


    method_list = [params.method]


    for ood_name in OOD_DATASET[dataset_name]:
        for method in method_list:
            if method in ["feature-based", "verbalized", "verbalized-self", "verbalized-iterative",
                          "verbalized-multitask", 'verbalized-iterative-ablation', 'verbalized-iterative-ablation2', 'LCF','consistent'] or 'LCF' in method:
                method_no_suffix = method
                method = method + "-calibration"
            # seeds = 1 if method != "ensemble" else 5
            seeds=params.seed
            alllabels_list = []
            allprobs_list = []
            allpreds_list = []
            if single:
                seed = seeds
                # for seed in range(seeds):
                if "-calibration" in method:
                        print("i am here.")
                        allprobs = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy").tolist()
                        allpreds = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method_no_suffix}/{ood_name}/{seed}/allpreds.npy").tolist()
                        alllabels = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method_no_suffix}/{ood_name}/{seed}/alllabels.npy").tolist()
                else:
                        allprobs = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy").tolist()
                        allpreds = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allpreds.npy").tolist()
                        alllabels = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/alllabels.npy").tolist()

                alllabels_list.append(alllabels)
                allprobs_list.append(allprobs)
                allpreds_list.append(allpreds)

                compute_ece(allprobs_list, allpreds_list, alllabels_list)
                method = method.replace("-calibration", '')
                # method = method.strip("-calibration")
                if method in ["feature-based", "verbalized", "verbalized-self", "verbalized-iterative",
                              "verbalized-multitask", 'verbalized-iterative-ablation',
                              'verbalized-iterative-ablation2']:
                    if dataset_name == "dynasent":
                        continue
                    acc_list = []
                    for seed in range(seeds):
                        acc = np.load(
                            f"./results/metrics/ood/{dataset_name}/accuracy_of_{method}/{model_name}/{ood_name}-{seed}.npy")
                        acc_list.append(acc.item())
                    avg_acc = np.mean(acc_list)
                    print(avg_acc)
                    with open(f"./results/metrics/ood/{dataset_name}/{model_name}.tsv", "r") as f:
                        data = []
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip().split("\t")
                            data.append(line)
                        data[-1][0] = str(avg_acc)
                    with open(f"./results/metrics/ood/{dataset_name}/{model_name}.tsv", "w") as f:
                        for line in data:
                            print("\t".join(line), file=f)

            else:
                for seed in range(seeds):
                    if "-calibration" in method:
                        print("i am here.")
                        allprobs = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy").tolist()
                        allpreds = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method_no_suffix}/{ood_name}/{seed}/allpreds.npy").tolist()
                        alllabels = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method_no_suffix}/{ood_name}/{seed}/alllabels.npy").tolist()
                    else:
                        allprobs = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allprobs.npy").tolist()
                        allpreds = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/allpreds.npy").tolist()
                        alllabels = np.load(
                            f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_name}/{seed}/alllabels.npy").tolist()

                    alllabels_list.append(alllabels)
                    allprobs_list.append(allprobs)
                    allpreds_list.append(allpreds)

                compute_ece(allprobs_list, allpreds_list, alllabels_list)
                method = method.replace("-calibration",'')
                # method = method.strip("-calibration")
                if method in ["feature-based", "verbalized", "verbalized-self", "verbalized-iterative",
                              "verbalized-multitask", 'verbalized-iterative-ablation', 'verbalized-iterative-ablation2']:
                    if dataset_name == "dynasent":
                        continue
                    acc_list = []
                    for seed in range(seeds):
                        acc = np.load(
                            f"./results/metrics/ood/{dataset_name}/accuracy_of_{method}/{model_name}/{ood_name}-{seed}.npy")
                        acc_list.append(acc.item())
                    avg_acc = np.mean(acc_list)
                    print(avg_acc)
                    with open(f"./results/metrics/ood/{dataset_name}/{model_name}.tsv", "r") as f:
                        data = []
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip().split("\t")
                            data.append(line)
                        data[-1][0] = str(avg_acc)
                    with open(f"./results/metrics/ood/{dataset_name}/{model_name}.tsv", "w") as f:
                        for line in data:
                            print("\t".join(line), file=f)


def entropy():
    OOD_DATASET = {
        "sst2": ["sst2_iid", "bookcorpus", "random_words"],
        "yahoo_answers_topics": ["yahoo_answers_topics_iid", "bookcorpus", "random_words"]
    }
    os.makedirs(f"./results/metrics/ood/{dataset_name}/entropy", exist_ok=True)
    with open(f"./results/metrics/ood/{dataset_name}/entropy/{model_name}.tsv", "w") as f:
        f.write("method\tavg_prob\tavg_entropy\n")
        for ood_dataset in OOD_DATASET[dataset_name]:
            for method in ["None", "temperature_scaling", "label_smoothing", "eda", "ensemble",
                           "feature-based", "verbalized", "verbalized-self", "verbalized-iterative",
                           "verbalized-multitask"]:
                if method in ["feature-based", "verbalized", "verbalized-self", "verbalized-iterative",
                              "verbalized-multitask"]:
                    method = method + "-calibration"
                # print(f"method: {method}, domain: {ood_name}")
                seeds = 1 if method != "ensemble" else 5
                allprobs_list = []
                allentropy_list = []
                for seed in range(seeds):
                    allprobs = np.load(
                        f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_dataset}/{seed}/allprobs.npy").tolist()
                    allentropy = np.load(
                        f"./results/ood/{dataset_name}/{model_name}/{method}/{ood_dataset}/{seed}/allentropy.npy").tolist()

                    allprobs_list.append(allprobs)
                    allentropy_list.append(allentropy)
                avg_prob = np.mean(allprobs_list)
                avg_entropy = np.mean(allentropy_list)
                if "-calibration" in method:
                    method = method.rstrip("-calibration")
                f.write(f"{method}\t{avg_prob}\t{avg_entropy}\n")


COMPUTE = {
    "pretrain": pretrain,
    "kshots": kshots,
    "single_model": single_model,
    "dynamics": dynamics,
    "scale": scale,
    "parameter": parameter,
    "soft_prompt": soft_prompt,
    "ood": ood,
    "entropy": entropy
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='t5')
    parser.add_argument('--dataset_name', default='amazon_food')
    parser.add_argument('--single', default='False')
    params = parser.parse_args()

    # seed = params.seed
    single = eval(params.single)

    # for model_name in ["t5-base-middle", "t5-base-small", "t5-large-middle", "t5-small-middle"]:
    for model_name in [params.model_name]:
        for setting in ["ood"]:
            for dataset_name in [params.dataset_name]:

                if dataset_name == "yahoo":
                    dataset_name = "yahoo_answers_topics"
                if model_name == "roberta" and setting == "scale":
                    model_name = "bert"
                # print(model_name, dataset_name, '\n')

                result_path = f"./results/metrics/{setting}/{dataset_name}"
                os.makedirs(result_path, exist_ok=True)
                if setting != "entropy":
                    with open(os.path.join(result_path, f"{model_name}.tsv"), "w") as f:
                        print(
                            "acc\tavg_probs\t|avg_acc-avg_prob|\tECE\tECE on True\tECE on False\tECE_mass\tECE_mass on True\tECE_mass on False\tstd_acc\tstd__probs\t|std_acc-std_prob|\tstd_ECE\tstd_ECE on True\tstd_ECE on False\tstd_ECE_mass\tstd_ECE_mass on True\tstd_ECE_mass on False",
                            file=f)
                COMPUTE[setting]()
                # f.close()


