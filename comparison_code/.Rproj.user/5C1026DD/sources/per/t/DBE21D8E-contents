# Feature extraction functions
# Version: 1.1
# Author: Tom Geddes
# Changes:
# - Include replicate information for co-apex scores where available.

library("magrittr")
library("tidyverse")
source("featureextraction_v1.1.R")

library("caret")
library("MASS")
library("e1071")
library("class")
library("randomForest")

preddn = "predictions_v1.1"
resdn = "results_v1.1"

if (!dir.exists(preddn)) dir.create(preddn)
if (!dir.exists(resdn)) dir.create(resdn)

metadata = data.frame(fn = c(
    'datasets/rescaled/SEC_norm_iBAQ_rescaled.txt',
    'datasets/rescaled/SAX_norm_iBAQ_rescaled.txt',
    'datasets/rescaled/HIC_norm_iBAQ_rescaled.txt',
    'datasets/rescaled/HIC_NaCl_norm_iBAQ_rescaled.txt',
    'datasets/rescaled/CHAPS_rescaled.txt',
    'datasets/rescaled/Mann_rescaled.txt',
    'datasets/rescaled/Gygi_rescaled.txt',
    'datasets/rescaled/subcell_rescaled.txt',
    'datasets/rescaled/TCL_rescaled.txt',
    'datasets/rescaled/MOESM5_ESM_rescaled.txt'
), stringsAsFactors = FALSE)

metadata$pidfn = c(
    'datasets/SEC_norm_iBAQ_PId.txt',
    'datasets/SAX_norm_iBAQ_PId.txt',
    'datasets/HIC_norm_iBAQ_PId.txt',
    'datasets/HIC_NaCl_norm_iBAQ_PId.txt',
    'datasets/CHAPS_scaled_PId.txt',
    'datasets/Mann_scaled_PId.txt',
    'datasets/Gygi_scaled_PId.txt',
    'datasets/subcell_scaled_PId.txt',
    'datasets/TCL_scaled_PId.txt',
    'datasets/MOESM5_ESM_scaled_PId.txt'
)

metadata$structure = c(
    NA,
    NA,
    NA,
    NA,
    'datasets/groups/CHAPS_groups_nonorm.txt',
    NA,
    NA,
    NA,
    NA,
    NA
)

    
metadata$name = c(
    'SEC',
    'SAX',
    'HIC_data',
    'HIC_NaCl',
    'CHAPS',
    'Mann_data',
    'Gygi',
    'subcell',
    'TCL',
    'MOESM5'
)

metadata$header =  c(F, F, F, F, F, F, F, F, F, F)
metadata$pearson = c(T, T, T, T, T, T, T, T, T, F)
metadata$ncc    =  c(T, T, T, T, T, F, F, F, F, F)
metadata$coapex  = c(T, T, T, T, T, F, F, F, F, F)
metadata$impute0 = c(F, F, F, F, T, T, T, T, T, T)

file_order = order(file.size(metadata$fn))
already_done = c()
file_order = file_order[!metadata$name[file_order] %in% already_done]

include_raw = TRUE
seeds = 1:10
nfolds = 10
num_cores = rep(10, 10)
# num_cores[file_order] = c(rep(10, 8), rep(5, 2))

gold_standard = readr::read_tsv("datasets/big_gold_standard.txt", col_names = FALSE)
gold_standard = gold_standard[sample(1:nrow(gold_standard), size = nrow(gold_standard), replace = FALSE),]
gold_standard %<>% rename(Protein_A = X1, Protein_B = X2) %>% filter(Protein_A != Protein_B)

lapply(file_order, function (dataset_i) {
    md = metadata[dataset_i,]
    cat("Processing dataset ", md$name, "...\n", sep = "")
    
    dat = readr::read_delim(md$fn, delim = "\t", col_names = md$header)
    dat_pids = readr::read_delim(md$pidfn, delim = "\t", col_names = FALSE) %>% pull(X1)
    retained_rows = dat %>% dplyr::select(where(is.numeric)) %>% apply(1, function(row) any(row != 0))
    
    dat %<>% filter(retained_rows)
    dat_pids = dat_pids[retained_rows]
    
    if (!is.na(md$structure)) {
        dat_structure = readr::read_delim(md$structure, delim = "\t", col_names = TRUE)
    } else {
        dat_structure = NULL
    }
    
    retained_rows = sample(seq_along(dat_pids), size = 1000, replace = FALSE) %>% is.element(seq_along(dat_pids), .)
    
    dat %<>% filter(retained_rows)
    dat_pids = dat_pids[retained_rows]
    
    mclapply(seeds, function (seed) {
        cat(md$name, "seed:", seed, " at ", format(Sys.time(), "%a %b %d %X"), "\n")
        
        folds = generate_folds(dat_pids, gold_standard, nfolds, seed)
        
        lapply(seq(nfolds), function(fold) {
            
            gs = bind_rows(folds$gs_groups[-fold])
            np = do.call(c, folds$negative_proteins[-fold])
            interactions = generate_interactions(gs, np, seed)
            
            training_pids = interactions$pids
            training_set = generate_featureset(training_pids, dat, dat_pids,
                                               seed = seed,
                                               impute0 = md$impute0,
                                               include_raw = include_raw,
                                               pearson = md$pearson,
                                               ncc = md$ncc,
                                               coapex = md$coapex,
                                               coapex_structure = dat_structure)
            
            training_set$Class = interactions$labels
            
            gs = folds$gs_groups[[fold]]
            np = folds$negative_proteins[[fold]]
            interactions = generate_interactions(gs, np, seed)
            
            test_pids = interactions$pids
            test_set = generate_featureset(test_pids, dat, dat_pids,
                                           seed = seed,
                                           impute0 = md$impute0,
                                           include_raw = include_raw,
                                           pearson = md$pearson,
                                           ncc = md$ncc,
                                           coapex = md$coapex,
                                           coapex_structure = dat_structure)
            
            test_labels = interactions$labels
            
            confusion = tibble(Method = character(), TP = integer(), TN = integer(), FP = integer(), FN = integer())
            fold_predictions = tibble(Seed = seed, Class = test_labels)
            
            # Probability predictions (regressions) for ROC curves
            # KNN
            preds = knn(train = dplyr::select(training_set, !Class), test = test_set, cl = training_set$Class, k=5, prob = TRUE)
            probs = attr(preds, which = "prob")
            preds %<>% as.character %>% as.numeric
            fold_predictions$knn = preds * probs + (1 - preds) * (1 - probs)
            
            # Logistic Regerssion       
            lr.model = glm(Class ~ ., data = training_set, family = binomial(link="logit"))
            fold_predictions$glm = predict(lr.model, test_set, type = "response")
            
            # SVM
            svm.model = svm(Class~., data = training_set)
            fold_predictions$svm = predict(svm.model, test_set)
            
            # Random Forest
            rf.model = randomForest(Class ~ ., data = training_set)
            fold_predictions$rf = predict(rf.model, test_set)
            
            
            # Class predictions for accuracy metrics
            training_set$Class %<>% factor(levels = c(0, 1))
            
            # KNN
            preds = knn(train = training_set[, colnames(training_set) != "Class"], test = test_set, cl = training_set$Class, k=5)
            confusion %<>% bind_rows(tibble(
                Method = "knn",
                TP = sum(test_labels == 1 & preds == 1),
                TN = sum(test_labels == 0 & preds == 0),
                FP = sum(test_labels == 0 & preds == 1),
                FN = sum(test_labels == 1 & preds == 0)
            ))
            
            # Logistic Regerssion
            lr.model = glm(Class ~ ., data = training_set, family = binomial(link="logit")) 
            pred.probs = predict(lr.model, test_set, type = "response")
            preds = ifelse(pred.probs > 0.5, 1, 0)
            confusion %<>% bind_rows(tibble(
                Method = "glm",
                TP = sum(test_labels == 1 & preds == 1),
                TN = sum(test_labels == 0 & preds == 0),
                FP = sum(test_labels == 0 & preds == 1),
                FN = sum(test_labels == 1 & preds == 0)
            ))
            
            # SVM
            svm.model = svm(Class~., data = training_set)
            preds = predict(svm.model, test_set)
            confusion %<>% bind_rows(tibble(
                Method = "svm",
                TP = sum(test_labels == 1 & preds == 1),
                TN = sum(test_labels == 0 & preds == 0),
                FP = sum(test_labels == 0 & preds == 1),
                FN = sum(test_labels == 1 & preds == 0)
            ))
            
            # Random Forest
            rf.model = randomForest(Class ~ ., data = training_set)
            preds = predict(rf.model, test_set)
            confusion %<>% bind_rows(tibble(
                Method = "rf",
                TP = sum(test_labels == 1 & preds == 1),
                TN = sum(test_labels == 0 & preds == 0),
                FP = sum(test_labels == 0 & preds == 1),
                FN = sum(test_labels == 1 & preds == 0)
            ))
            
            tibble(Confusion = list(confusion), Predictions = list(fold_predictions)) %>% return()
        }) %>% bind_rows -> seed_output
        
        
        metrics = seed_output$Confusion %>% bind_rows %>% group_by(Method) %>% summarise(
            TP = sum(TP),
            TN = sum(TN),
            FP = sum(FP),
            FN = sum(FN)
        ) %>% transmute(
            Dataset = md$name,
            Method = Method,
            Sensitivity = TP / (TP + FN),
            Specificity = TN / (TN + FP)
            #F1 = TP / (TP + (FP + FN) / 2)
        ) %>% mutate(
            Accuracy = (Sensitivity + Specificity) / 2
        )
        
        predictions = seed_output$Predictions %>% bind_rows
        tibble(Metrics = list(metrics), Predictions = list(predictions)) %>% return()
    
    }, mc.cores = num_cores[dataset_i]) %>% bind_rows -> dataset_output
    
    
    metrics = dataset_output$Metrics %>% bind_rows %>%
        group_by(Dataset, Method) %>%
        summarise(
            Sensitivity_mean = mean(Sensitivity),
            Specificity_mean = mean(Specificity),
            Accuracy_mean = mean(Accuracy),
            
            Sensitivity_sd = sd(Sensitivity),
            Specificity_sd = sd(Specificity),
            Accuracy_sd = sd(Accuracy),
            
            num_replicates = n()
        ) %>% ungroup
    
    predictions = dataset_output$Predictions %>% bind_rows
    
    dataset_output$Metrics %>% bind_rows %>% readr::write_csv(path = paste0(resdn, "/", md$name, "_seed_results.csv"))
    metrics %>% readr::write_csv(path = paste0(resdn, "/", md$name, "_summary_results.csv"))
    predictions %>% readr::write_csv(path = paste0(preddn, "/", md$name, "_seed_predictions.csv"))
    
    tibble(Metrics = list(metrics),
           Predictions = list(predictions)) %>% return()
}) %>% bind_rows -> final_output


final_output$Metrics %>% bind_rows %>% readr::write_csv(path = paste0(resdn, "/all_results.csv"))
final_output$Predictions %>% bind_rows %>% readr::write_csv(path = paste0(preddn, "/all_predictions.csv"))