import numpy as np
from keras import backend as K
from keras import layers as L
from keras import regularizers as R
from keras import optimizers as O
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from data_generation import make_interaction_generator, PredictiveSequence, all_pairs
from data_generation import filter_empty_rows, save_predictions, save_results
from math import ceil
import os
from prettytable import PrettyTable
import argparse
import csv

K.set_floatx("float32")

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seed', type=int, default=None, help="Random seed.")
parser.add_argument('-t', '--testset', type=str, default="dev")
parser.add_argument('-a', '--allpreds', action='store_true', default=False)
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

seed = args.seed
individual_batch_size = 16
ensemble_batch_size = 16
num_workers = 1
dev_or_test = args.testset
feature_space = 16
max_epochs = 1000
generate_all_predictions = args.allpreds


print("Finished importing libraries.")
print("Using seed ", seed, " and testing against the ", dev_or_test, " set.", sep="")
# Dataset information
set_names = [
    "HIC_NaCl_norm_iBAQ",
    "CHAPS",
    "Gygi",
    "HIC_norm_iBAQ",
    "SEC_norm_iBAQ",
    "Mann",
    "SAX_norm_iBAQ",
    "Subcell",
    "TCL",
    "MOESMS_ESM"
]
pid_filenames = [
    "datasets/HIC_NaCl_norm_iBAQ_PId.txt",
    "datasets/CHAPS_scaled_PId.txt",
    "datasets/Gygi_scaled_PId.txt",
    "datasets/HIC_norm_iBAQ_PId.txt",
    "datasets/SEC_norm_iBAQ_PId.txt",
    "datasets/Mann_scaled_PId.txt",
    "datasets/SAX_norm_iBAQ_PId.txt",
    "datasets/subcell_scaled_PId.txt",
    "datasets/TCL_scaled_PId.txt",
    "datasets/MOESM5_ESM_scaled_PId.txt"
]

data_filenames = [
    "datasets/filtered/filtered_HIC_NaCl_with_names.txt",
    "datasets/filtered/filtered_chaps_averages.txt",
    "datasets/filtered/filtered_Gygi_with_names.txt",
    "datasets/filtered/filtered_HIC_data_with_names.txt",
    "datasets/filtered/filtered_sec_averages.txt",
    "datasets/filtered/filtered_Mann_data_with_names.txt",
    "datasets/filtered/filtered_SAX_data_with_names.txt",
    "datasets/filtered/filtered_subcell_with_names.txt",
    "datasets/filtered/filtered_TCL_with_names.txt",
    "datasets/filtered/filtered_MOESM5_with_names.txt"
]


def get_widths (input_width):
    return [ceil(input_width / 2)] + [feature_space]


def calculate_metrics (prediction_class, true_class):
    TP = np.sum(np.logical_and(true_class == 1, prediction_class == 1), dtype=np.float32)
    TN = np.sum(np.logical_and(true_class == 0, prediction_class == 0), dtype=np.float32)
    FP = np.sum(np.logical_and(true_class == 0, prediction_class == 1), dtype=np.float32)
    FN = np.sum(np.logical_and(true_class == 1, prediction_class == 0), dtype=np.float32)

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = np.mean(prediction_class == true_class, dtype=np.float32)

    return sensitivity, specificity, accuracy


# Load and filter data
print("Loading protein data.")
# dataset_list = [np.loadtxt(fn, dtype=np.float32, delimiter="\t") for fn in data_filenames]
# pid_list = [np.loadtxt(fn, dtype=str, delimiter="\t").flatten() for fn in pid_filenames]
# Loading prefiltered data for comparison
converter = {0: lambda s: 0.}
dataset_list = [np.loadtxt(fn, dtype=np.float32, delimiter="\t", converters=converter)[:,1:] for fn in data_filenames]
pid_list = [np.loadtxt(fn, dtype=str, delimiter="\t", usecols=0) for fn in data_filenames]
gold_standard = np.loadtxt("datasets/positives.txt", dtype=str, delimiter="\t")

print("Filtering empty rows.")
for i in range(len(dataset_list)):
    dataset_list[i], pid_list[i] = filter_empty_rows(dataset_list[i], pid_list[i])

ordered_list = [False] * len(set_names)
ensemble_model_widths = [ceil(len(set_names) * feature_space / 2)] + [feature_space] + [2]

# Generate interaction sets
print("Generating interactions...", end=" ")

interaction_generator = make_interaction_generator(pid_list,
                                                   gold_standard,
                                                   return_individual=True,
                                                   split=[18, 1, 1],
                                                   split_names=["train", "dev", "test"],
                                                   seed=seed)

ensemble_interaction_pids, ensemble_interaction_labels = next(interaction_generator)
individual_interaction_pids = []
individual_interaction_labels = []

print("Done.")

subset_names = ["train", "dev", "test"]

# Construct individual models
print("Constructing models.")

individual_models = []
individual_inputs = []
observation_multipliers = []

K.clear_session()

dataset_zip = zip(
    set_names,
    dataset_list,
    pid_list,
    interaction_generator,
    ordered_list
)

individual_sensitivity = []
individual_specificity = []
individual_accuracy = []

required_directories = [
    "test_predictions",
    "all_predictions",
    "results",
    "representations",
]

for d in required_directories:
    if not os.path.isdir(d):
        os.mkdir(d)

print("Constructing individual models.")
for name, dataset, pids, int_data, ordered in dataset_zip:
    int_pids, int_labs = int_data
    individual_interaction_pids.append(int_pids)
    individual_interaction_labels.append(int_labs)

    print("Constructing model for dataset ", name, ".", sep="")

    tns = individual_input = L.Input(shape=(2, dataset.shape[1]))
    tns = L.Conv1D(filters=10,
                   kernel_size=1,
                   padding="same",
                   activation='relu',
                   kernel_regularizer=R.l2(0.001))(tns)

    tns = L.Flatten()(tns)
    widths = get_widths(dataset.shape[1])
    for _, width in enumerate(widths[:-1]):
        tns = L.Dense(width, activation='relu', kernel_regularizer=R.l2(0.001))(tns)

    tns = L.Dense(widths[-1], activation='tanh', kernel_regularizer=R.l2(0.001), activity_regularizer=R.l2(0.001))(tns)

    individual_model = Model(inputs=[individual_input], outputs=[tns])

    individual_models.append(individual_model)
    individual_inputs.append(individual_input)
    individual_output = L.Dense(2, activation='softmax')(tns)
    pretrain_model = Model(inputs=[individual_input], outputs=[individual_output])
    pretrain_model.compile(optimizer=O.Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])

    print("\tGenerating interaction sequences")
    train_sequence = PredictiveSequence(pid_list=[pids],
                                        mat_list=[dataset],
                                        intarr=int_pids["train"],
                                        labarr=int_labs["train"],
                                        batch_size=individual_batch_size,
                                        seed=seed)

    development_sequence = PredictiveSequence(pid_list=[pids],
                                              mat_list=[dataset],
                                              intarr=int_pids["dev"],
                                              labarr=int_labs["dev"],
                                              batch_size=individual_batch_size,
                                              seed=seed,
                                              shuffle=False)

    test_sequence = PredictiveSequence(pid_list=[pids],
                                       mat_list=[dataset],
                                       intarr=int_pids[dev_or_test],
                                       labarr=int_labs[dev_or_test],
                                       batch_size=individual_batch_size,
                                       seed=seed,
                                       shuffle=False)

    model_path = "model_" + name + ".hdf5"
    loss_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20)
    save_callback = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)

    print("\tTraining prediction model...")
    pretrain_model.fit_generator(generator=train_sequence,
                                 epochs=max_epochs,
                                 verbose=1,
                                 validation_data=development_sequence,
                                 callbacks=[loss_callback, save_callback],
                                 workers=num_workers,
                                 use_multiprocessing=False)

    print("\tEvaluating prediction model manually")
    predictions = pretrain_model.predict_generator(test_sequence,
                                                   workers=num_workers,
                                                   use_multiprocessing=False)

    save_predictions(predictions[:, 1],
                     int_labs[dev_or_test][:, 1],
                     method="deep_individual",
                     path="test_predictions/" + name + "_individual_" + dev_or_test + "_s" + str(seed) + "_predictions.csv", seed=seed)

    prediction_class = np.argmax(predictions, axis=1)
    true_class = np.argmax(int_labs[dev_or_test], axis=1)

    sensitivity, specificity, accuracy = calculate_metrics(prediction_class, true_class)

    individual_sensitivity.append(round(sensitivity, 3))
    individual_specificity.append(round(specificity, 3))
    individual_accuracy.append(round(accuracy, 3))

    print("\tSensitivity:", round(sensitivity, 3), "\tSelectivity:", round(specificity, 3), "\tAccuracy:", round(accuracy, 3))

    if generate_all_predictions:
        print("Generating " + name + " predictions for all pairs...")

        num_complete = 0
        total = len(pids) * (len(pids) - 1) / 2
        batch_size = 1024

        pair_generator = all_pairs([pids], [dataset], batch_size=batch_size)
        prediction_path = "all_predictions/" + name + "_individual_predictions_s" + str(seed) + ".csv"

        with open(prediction_path, mode='w') as prediction_file:
            prediction_writer = csv.writer(prediction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            header_data = ["Protein_A", "Protein_B"]
            header_data += ["Forward_Score", "Reverse_Score", "Combined"]
            prediction_writer.writerow(header_data)

            for left_pids, right_pids, batch_inputs in pair_generator:

                predictions = pretrain_model.predict_on_batch(batch_inputs)
                forward_scores = predictions[:, 1]

                batch_inputs = [np.flip(bi, axis=1) for bi in batch_inputs]
                predictions = pretrain_model.predict_on_batch(batch_inputs)
                reverse_scores = predictions[:, 1]

                combined_scores = np.stack([forward_scores, reverse_scores]).mean(axis=0)

                for i in range(len(left_pids)):
                    row_data = [left_pids[i], right_pids[i]]
                    row_data += [forward_scores[i], reverse_scores[i], combined_scores[i]]

                    prediction_writer.writerow(row_data)

                num_complete += batch_size
                print('{:.2f}'.format(round(100 * num_complete / total, 2)), "% complete.", end="\r")
            print()

    print("\tDone.", end="\n\n")

    del train_sequence
    del development_sequence
    del test_sequence
    del predictions

print("Individual models constructed. Constructing ensemble model...")

# Construct ensemble (second step) model
feature_width = feature_space * len(dataset_list)
tns = ensemble_input = L.Input(shape=(feature_width,))

for _, width in enumerate(ensemble_model_widths[:-1]):
    tns = L.Dense(width, activation='relu', kernel_regularizer=R.l2(0.01))(tns)

tns = L.Dense(ensemble_model_widths[-1], activation='softmax')(tns)

ensemble_model = Model(inputs=[ensemble_input], outputs=[tns])


# Construct combined model
tns = [i_model(i_input) for i_model, i_input in zip(individual_models, individual_inputs)]
observation_multipliers = [L.Input(shape=(1,)) for i in range(len(individual_inputs))]
tns = [L.multiply([individual_tns, observed]) for individual_tns, observed in zip(tns, observation_multipliers)]

ensemble_features = L.Concatenate()(tns)
tns = ensemble_model(ensemble_features)

combined_model = Model(inputs=individual_inputs + observation_multipliers, outputs=[tns])
combined_model.compile(optimizer=O.Adam(), loss=['binary_crossentropy'], metrics=['accuracy'])

train_sequence = PredictiveSequence(intarr=ensemble_interaction_pids["train"],
                                    labarr=ensemble_interaction_labels["train"],
                                    pid_list=pid_list,
                                    mat_list=dataset_list,
                                    batch_size=ensemble_batch_size,
                                    seed=seed)

development_sequence = PredictiveSequence(intarr=ensemble_interaction_pids["dev"],
                                          labarr=ensemble_interaction_labels["dev"],
                                          pid_list=pid_list,
                                          mat_list=dataset_list,
                                          batch_size=ensemble_batch_size,
                                          seed=seed,
                                          shuffle=False)

test_sequence = PredictiveSequence(intarr=ensemble_interaction_pids[dev_or_test],
                                   labarr=ensemble_interaction_labels[dev_or_test],
                                   pid_list=pid_list,
                                   mat_list=dataset_list,
                                   batch_size=ensemble_batch_size,
                                   seed=seed,
                                   shuffle=False)

model_path = "model_ensemble.hdf5"
loss_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20)
save_callback = ModelCheckpoint(filepath=model_path, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)

# Train combined model

print("Training combined model.", end=" ")
combined_model.fit_generator(generator=train_sequence,
                             epochs=max_epochs,
                             callbacks=[loss_callback, save_callback],
                             verbose=1,
                             validation_data=development_sequence,
                             workers=num_workers,
                             use_multiprocessing=False)

print("Done.")

print("Reloading model weights from best epoch.")
combined_model.load_weights(filepath=model_path)

print("Evaluating...")
predictions = combined_model.predict_generator(test_sequence,
                                               workers=num_workers,
                                               use_multiprocessing=False)

save_predictions(predictions[:, 1],
                 ensemble_interaction_labels[dev_or_test][:, 1],
                 method="deep_ensemble",
                 path="test_predictions/all_ensemble_" + dev_or_test + "_s" + str(seed) + "_predictions.csv",
                 seed=seed)

prediction_class = np.argmax(predictions, axis=1)
true_class = np.argmax(ensemble_interaction_labels[dev_or_test], axis=1)
sensitivity, specificity, accuracy = calculate_metrics(prediction_class, true_class)

print("\tSensitivity:", round(sensitivity, 3), "\tSelectivity:", round(specificity, 3), "\tAccuracy:", round(accuracy, 3), end="\n\n")

ensemble_dataset_sensitivity = []
ensemble_dataset_specificity = []
ensemble_dataset_accuracy= []

print("Testing combined model on individual interaction sets... please wait.", end="\n\n")

dataset_zip = zip(
    set_names,
    dataset_list,
    pid_list,
    individual_interaction_pids,
    individual_interaction_labels
)

for name, dataset, widths, int_pids, int_labs in dataset_zip:
    print("Evaluating full model on ", name, "... ", sep="", end="")

    test_sequence = PredictiveSequence(pid_list=pid_list,
                                       mat_list=dataset_list,
                                       intarr=int_pids[dev_or_test],
                                       labarr=int_labs[dev_or_test],
                                       batch_size=ensemble_batch_size,
                                       seed=seed,
                                       shuffle=False)

    predictions = combined_model.predict_generator(test_sequence,
                                                      workers=num_workers,
                                                      use_multiprocessing=False)

    save_predictions(predictions[:,1],
                     test_sequence.get_all_labels()[:,1],
                     method="deep_ensemble",
                     path="test_predictions/" + name + "_ensemble_" + dev_or_test + "_s" + str(seed) + "_predictions.csv",
                     seed=seed)

    prediction_class = np.argmax(predictions, axis=1)
    true_class = np.argmax(test_sequence.get_all_labels(), axis=1)

    sensitivity, specificity, accuracy = calculate_metrics(prediction_class, true_class)

    ensemble_dataset_sensitivity.append(round(sensitivity, 3))
    ensemble_dataset_specificity.append(round(specificity, 3))
    ensemble_dataset_accuracy.append(round(accuracy, 3))

    print("Done.", end="\n")


bar_width = 1 / 3

# plt.subplot(1, 3, 1)
# plt.bar([i for i, _ in enumerate(individual_sensitivity)], individual_sensitivity, width=bar_width)
# plt.bar([i + bar_width for i, _ in enumerate(individual_sensitivity)], ensemble_dataset_sensitivity, width=bar_width)
#
# plt.subplot(1, 3, 2)
# plt.bar([i for i, _ in enumerate(individual_specificity)], individual_specificity, width=bar_width)
# plt.bar([i + bar_width for i, _ in enumerate(individual_specificity)], ensemble_dataset_specificity, width=bar_width)
#
# plt.subplot(1, 3, 3)
# plt.bar([i for i, _ in enumerate(individual_specificity)], individual_specificity, width=bar_width)
# plt.bar([i + bar_width for i, _ in enumerate(individual_accuracy)], ensemble_dataset_accuracy, width=bar_width)
#
# plt.savefig("comparison.png")

tbl = PrettyTable()
tbl.field_names = ["Dataset",
                   "Ind. sensitivity", "Ens. sensitivity",
                   "Ind. specificity", "Ens. specificity",
                   "Ind. accuracy", "Ens. accuracy"]

for fn in tbl.field_names:
    tbl.align[fn] = "l"

metrics_zip = zip(
    set_names,
    individual_sensitivity,
    ensemble_dataset_sensitivity,
    individual_specificity,
    ensemble_dataset_specificity,
    individual_accuracy,
    ensemble_dataset_accuracy
)

for row_data in metrics_zip:
    tbl.add_row(row_data)

tbl.add_row([
    "Mean",
    round(sum(individual_sensitivity) / len(set_names), 3),
    round(sum(ensemble_dataset_sensitivity) / len(set_names), 3),
    round(sum(individual_specificity) / len(set_names), 3),
    round(sum(ensemble_dataset_specificity) / len(set_names), 3),
    round(sum(individual_accuracy) / len(set_names), 3),
    round(sum(ensemble_dataset_accuracy) / len(set_names), 3)
])
print(tbl)

save_results(
    sensitivity=individual_sensitivity + ensemble_dataset_sensitivity,
    specificity=individual_specificity + ensemble_dataset_specificity,
    accuracy=individual_accuracy + ensemble_dataset_accuracy,
    dataset=set_names*2,
    method=["deep_individual"] * 10 + ["deep_ensemble"] * 10,
    path="results/ensemble_" + dev_or_test + "_s" + str(seed) + "_results.csv"
)

del train_sequence
del development_sequence
del test_sequence
del predictions

if generate_all_predictions:
    print("Generating ensemble predictions for all pairs...")
    num_complete = 0
    batch_size = 1024

    pair_generator = all_pairs(pid_list, dataset_list, batch_size=batch_size)
    prediction_path = "all_predictions/all_ensemble_predictions_s" + str(seed) + ".csv"

    with open(prediction_path, mode='w') as prediction_file:
        prediction_writer = csv.writer(prediction_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        header_data = ["Protein_A", "Protein_B"]
        header_data += [name + "_obs" for name in set_names]
        header_data += ["Forward_Score", "Reverse_Score", "Combined"]
        prediction_writer.writerow(header_data)

        for left_pids, right_pids, batch_inputs, batch_obs in pair_generator:

            input_data = batch_inputs + batch_obs
            predictions  = combined_model.predict_on_batch(input_data)
            forward_scores = predictions[:, 1]

            batch_inputs = [np.flip(bi, axis=1) for bi in batch_inputs]
            input_data = batch_inputs + batch_obs
            predictions = combined_model.predict_on_batch(input_data)
            reverse_scores = predictions[:, 1]

            combined_scores = np.stack([forward_scores, reverse_scores]).mean(axis=0)

            for i in range(len(left_pids)):
                row_data = [left_pids[i], right_pids[i]]
                row_data += [obs[i] for obs in batch_obs]
                row_data += [forward_scores[i], reverse_scores[i], combined_scores[i]]

                prediction_writer.writerow(row_data)

            num_complete += batch_size
            print(num_complete, "complete.", end="\r")
        print()

print("Done. Exiting.")
