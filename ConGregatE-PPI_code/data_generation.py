from math import ceil
import numpy as np
from keras.utils import Sequence
import csv

def filter_empty_rows(mat, pids=None):
    retain_rows = np.any(mat > 0., axis=1)
    return mat[retain_rows] if pids is None else mat[retain_rows], pids[retain_rows]


def make_interaction_generator(pid_list, gold_standard, return_individual=False, split=[18, 1, 1], split_names=None, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Make unique master pid array, unique gold standard protein array
    all_pids = np.unique(np.concatenate(pid_list))

    gold_pids = np.unique(gold_standard)
    negative_pids = np.setdiff1d(all_pids, gold_pids, assume_unique=True)
    del gold_pids

    # Find gold standard pairs present in any dataset
    gold_in_pids = [np.all(np.isin(gold_standard, pids), axis=1) for pids in pid_list]
    gold_in_pids = np.stack(gold_in_pids, axis=-1)
    gold_in_pids = np.any(gold_in_pids, axis=1)

    # Create positive interactions from gold standard pairs in data
    all_positive_links = gold_standard[gold_in_pids]

    # Convert all PIDs to integer indices to save on memory and processing time
    pid_dict = {key: idx for idx, key in enumerate(all_pids)}
    pid2idx = np.vectorize(lambda x: pid_dict[x])
    pid_list = [pid2idx(pids) for pids in pid_list]
    all_positive_links = pid2idx(all_positive_links)
    negative_pids = pid2idx(negative_pids)

    # Shuffle positive interactions, negative proteins
    np.random.shuffle(all_positive_links)
    np.random.shuffle(negative_pids)

    # Split positive interactions, negative proteins
    split = np.asarray(split, dtype=int)
    split_cum = split.cumsum()[:-1]

    split_breaks = (all_positive_links.shape[0] * split_cum / split.sum()).astype(int)
    positive_link_groups = np.split(all_positive_links, split_breaks)

    split_breaks = (negative_pids.shape[0] * split_cum / split.sum()).astype(int)
    negative_pid_groups = np.split(negative_pids, split_breaks)

    # Reusable function to make interaction data
    def make_set_interactions(positive_link_groups, negative_pid_groups, pid_list):

        # # Augment positive interactions by mirroring
        # for i, positive_links in enumerate(positive_link_groups):
        #     positive_link_groups[i] = np.concatenate([positive_links, positive_links[::-1]])

        interaction_pids = []
        interaction_labels = []

        # Generate negative interactions for each group
        for i, positive_links in enumerate(positive_link_groups):
            pids_from_positive = positive_links.flatten()

            pid_search = {
                posid: np.unique(np.concatenate([
                    pid_list[j].view() for j in
                    np.flatnonzero(np.asarray([np.isin(posid, pids) for k, pids in enumerate(pid_list)]))
                ])) for posid in np.unique(pids_from_positive)} if type(pid_list) is list else None

            negative_links = []

            for j, pid_from_positive in enumerate(pids_from_positive):
                while True:
                    pid_from_negative = np.random.choice(negative_pid_groups[i], 1)
                    proposed_link = np.asarray([pid_from_positive, pid_from_negative], dtype=int)
                    if type(pid_list) is list:
                        proposed_in_pids = np.isin(pid_from_negative, pid_search[pid_from_positive])
                    else:
                        proposed_in_pids = np.isin(pid_from_negative, pid_list)

                    if proposed_in_pids:
                        np.random.shuffle(proposed_link)
                        negative_links.append(proposed_link)
                        break

            negative_links = np.asarray(negative_links)

            # Augment positive interactions by mirroring
            positive_links = np.concatenate([positive_links, positive_links[::-1]])

            # Combine negative and positive interaction; append to group list
            combined_links = [positive_links.view(), negative_links.view()]
            interaction_pids.append(all_pids[np.concatenate(combined_links)])

            # Generate interaction labels; append to group list
            combined_labels = np.zeros((positive_links.shape[0] + negative_links.shape[0], 2))
            combined_labels[:positive_links.shape[0], 1] = 1.
            combined_labels[positive_links.shape[0]:, 0] = 1.
            interaction_labels.append(combined_labels)

            idx = np.arange(combined_labels.shape[0])
            np.random.shuffle(idx)
            interaction_pids[-1] = interaction_pids[-1][idx]
            interaction_labels[-1] = interaction_labels[-1][idx]

        return interaction_pids, interaction_labels

    ensemble_interaction_pids, ensemble_interaction_labels = make_set_interactions(
        positive_link_groups=positive_link_groups,
        negative_pid_groups=negative_pid_groups,
        pid_list=pid_list)

    if split_names is not None:
        ensemble_interaction_pids = {name: subset for name, subset in zip(split_names, ensemble_interaction_pids)}
        ensemble_interaction_labels = {name: subset for name, subset in zip(split_names, ensemble_interaction_labels)}

    # Construct interactions for individual models if desired
    if return_individual:
        yield ensemble_interaction_pids, ensemble_interaction_labels

        for idx, set_pids in enumerate(pid_list):
            filtered_positive_link_groups = [
                links[np.all(np.isin(links, set_pids), axis=1)] for i, links in enumerate(positive_link_groups)
            ]
            filtered_negative_pid_groups = [
                neg_pids[np.isin(neg_pids, set_pids)] for i, neg_pids in enumerate(negative_pid_groups)
            ]

            interaction_pids, interaction_labels = make_set_interactions(
                positive_link_groups=filtered_positive_link_groups,
                negative_pid_groups=filtered_negative_pid_groups,
                pid_list=set_pids
            )

            if split_names is not None:
                interaction_pids = {name: subset for name, subset in zip(split_names, interaction_pids)}
                interaction_labels = {name: subset for name, subset in zip(split_names, interaction_labels)}

            yield interaction_pids, interaction_labels
    else:
        return ensemble_interaction_pids, ensemble_interaction_labels


class PredictiveSequence(Sequence):

    def __init__(self, pid_list, mat_list, intarr, labarr=None, batch_size=16, seed=None, shuffle=True, convolutional=True):
        self.seed = seed
        self.intarr, self.labarr, self.pid_list, self.mat_list, self.shuffle = intarr, labarr, pid_list, mat_list, shuffle
        self.convolutional = convolutional
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(self.intarr.shape[0] / np.float32(self.batch_size)))

        for i, mat in enumerate(self.mat_list):
            np.append(mat_list[i], np.zeros(mat.shape[1]))

        all_pids = np.unique(np.concatenate([np.asarray(pids) for pids in pid_list]))
        self.dict_list = [{key: -1 for idx, key in enumerate(all_pids)} for _ in range(len(pid_list))]
        for i, pids in enumerate(pid_list):
            for j, key in enumerate(pids):
                self.dict_list[i][key] = j

        self.shuffle_data()

    def shuffle_data(self):
        idxarr = np.arange(self.intarr.shape[0])

        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(idxarr)
            self.seed = np.random.randint(low=2**16)

        self.epoch_intarr = self.intarr[idxarr]
        self.epoch_labarr = self.labarr[idxarr] if self.labarr is not None else None

    def on_epoch_end(self):
        self.shuffle_data()

    def __len__(self):
        return self.num_batches

    def __getitem__(self, batch_idx):
        this_batch_size = min([self.batch_size, self.intarr.shape[0] - batch_idx * self.batch_size])
        batch_int = self.epoch_intarr[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]
        batch_input_list = []
        batch_observed_list = []

        for idx_dict, pids, mat in zip(self.dict_list, self.pid_list, self.mat_list):
            getidx = np.vectorize(lambda x: idx_dict[x])
            idx = getidx(batch_int)
            if self.convolutional:
                batch_input_list.append(mat[idx].reshape(this_batch_size, 2, -1))
            else:
                batch_input_list.append(mat[idx].reshape(this_batch_size, -1))

            observed = np.logical_and(idx[:, 0] != -1, idx[:, 1] != -1).reshape(-1, 1)
            obs_mult = np.zeros(observed.shape)
            obs_mult[observed] = 1.
            if len(self.mat_list) > 1:
                batch_observed_list.append(obs_mult)

        if self.labarr is None:
            return batch_input_list + batch_observed_list
        else:
            batch_lab = self.epoch_labarr[(batch_idx * self.batch_size):((batch_idx + 1) * self.batch_size)]
            return batch_input_list + batch_observed_list, batch_lab

    def get_all_labels(self):
        return self.labarr.view()


# Batch generator for all pairs of proteins supported by data
def all_pairs(pid_list, mat_list, batch_size=16, convolutional=True):
    for i, mat in enumerate(mat_list):
        np.append(mat_list[i], np.zeros(mat.shape[1]))

    all_pids = np.unique(np.concatenate([np.asarray(pids) for pids in pid_list]))
    dict_list = [{key: -1 for idx, key in enumerate(all_pids)} for _ in range(len(pid_list))]
    for i, pids in enumerate(pid_list):
        for j, key in enumerate(pids):
            dict_list[i][key] = j

    batch_left_pid_list = []
    batch_right_pid_list = []
    batch_input_list = [[] for _ in pid_list]
    batch_observed_list = [[] for _ in pid_list]

    for i, left_pid in enumerate(all_pids):
        for right_pid in all_pids[i + 1:]:

            row_input_list = []
            row_observed_list = []

            for idx_dict, pids, mat in zip(dict_list, pid_list, mat_list):
                idx = np.array([idx_dict[left_pid], idx_dict[right_pid]])

                if convolutional:
                    row_input_list.append(mat[idx])
                else:
                    row_input_list.append(mat[idx].reshape(-1))

                observed = np.logical_and(idx[0] != -1, idx[1] != -1)
                obs_multiplier = 1. if observed else 0.

                row_observed_list.append(obs_multiplier)

            if max(row_observed_list) == 1.:
                batch_left_pid_list.append(left_pid)
                batch_right_pid_list.append(right_pid)

                for j, input_row, obs_multiplier in zip(range(len(row_input_list)), row_input_list, row_observed_list):
                    batch_input_list[j].append(input_row)
                    batch_observed_list[j].append(obs_multiplier)

            if len(batch_input_list[0]) >= batch_size:
                batch_input_list = [np.array(input_list) for input_list in batch_input_list]
                batch_observed_list = [np.array(obs_list) for obs_list in batch_observed_list]

                if len(pid_list) > 1:
                    yield batch_left_pid_list, batch_right_pid_list, batch_input_list, batch_observed_list
                else:
                    yield batch_left_pid_list, batch_right_pid_list, batch_input_list

                batch_input_list = [[] for _ in pid_list]
                batch_observed_list = [[] for _ in pid_list]
                batch_left_pid_list = []
                batch_right_pid_list = []

    if len(batch_input_list[0]) >= 0:
        batch_input_list = [np.array(input_list) for input_list in batch_input_list]
        batch_observed_list = [np.array(obs_list) for obs_list in batch_observed_list]

        if len(pid_list) > 1:
            yield batch_left_pid_list, batch_right_pid_list, batch_input_list, batch_observed_list
        else:
            yield batch_left_pid_list, batch_right_pid_list, batch_input_list


def save_predictions(predictions, labels, method, path, seed=None):
    with open(path, mode='w') as prediction_file:
        prediction_writer = csv.writer(prediction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        header_data = ["Class", method]
        if seed is not None:
            header_data = ["Seed"] + header_data
        prediction_writer.writerow(header_data)

        for prediction, label in zip(predictions, labels):
            row_data = [label, prediction]
            if seed is not None:
                row_data = [seed] + row_data

            prediction_writer.writerow(row_data)

def save_results(sensitivity, specificity, accuracy, dataset, method, path, seed=None):
    with open(path, mode='w') as results_file:
        results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        header_data = ["Dataset", "Method", "Sensitivity", "Specificity", "Accuracy"]
        if seed is not None:
            header_data = ["Seed"] + header_data
        results_writer.writerow(header_data)

        for ds, mth, sen, sel, acc in zip(dataset, method, sensitivity, specificity, accuracy):
            row_data = [ds, mth, sen, sel, acc]
            if seed is not None:
                row_data = [seed] + row_data

            results_writer.writerow(row_data)


def save_feature_map(feature_map, path):
    with open(path, mode='w') as fm_file:
        fm_writer = csv.writer(fm_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for fm_row in feature_map:
            fm_writer.writerow(fm_row.tolist())
