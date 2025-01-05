# Feature extraction functions
# Version: 1.1
# Author: Tom Geddes
# Changes:
# - Alter Co-apex score implementation to handle replicates when present
# - Alter handling of feature selection

#load in data files needed
library("magrittr")
library("tidyverse")
library("reshape2")
library("parallel")
library("impute")

####functions needed

#Cross-correlation function
LT.crosscor = function(x, y) {
  
  # arguments type & size should be checked
  n = length(x)
  
  # dummy square matrix to build shift matrix  
  A = matrix(0, n, n)
  
  # shift matrix
  shift = pmax(row(A)-col(A) + 1, 0) + 1
  
  # shift x and y
  shiftedX = matrix(c(0,x)[shift], n)
  shiftedY = matrix(c(0,y)[shift], n)
  
  max(y %*% shiftedX, x %*% shiftedY)
  
}

LT.NCC = function(x,y) {
  LT.crosscor(x,y)/max(crossprod(x), crossprod(y))
}

#this is the bit I actually use in my code:
get_ncc=function(x,raw_mat){
  a=x[1]
  b=x[2]
  ncc=LT.NCC(raw_mat[a,],raw_mat[b,])
  return(ncc)
}



#Co-apex stuff
repa = function(x){
  rep(x, (length(x)-1):0)
} 

repb = function(x){
  colb = c()
  for(i in 1:(length(x)-1)){
    colb = c(colb, x[(i+1):length(x)])
  }
  return (colb)
}

getapexpairs=function(x, coapex_structure = NULL){
  if (is.null(coapex_structure)) {
    coapex_structure = tibble(Replicate = 1, Fraction = seq(ncol(x)))
  }
  
  num_reps = coapex_structure$Replicate %>% unique %>% length # Get number of replicates
  x %<>% replace_na(0)
  
  coapex_scores = coapex_structure$Replicate %>% unique %>% lapply(function(r) { # By replicate
    apices = tibble( # Get column maxes
      idx = seq(nrow(x)),
      Apex = x[, coapex_structure$Replicate == r] %>% max.col(ties.method = "first")
    )
    
    share_lists = aggregate(idx ~ . , apices, FUN = list)$idx # Group proteins sharing apices
    share_lists = share_lists[lengths(share_lists) > 1] # Filter out apices with no pairs
    
    long24a = unlist(lapply(share_lists, repa)) # List pairs in long form
    long24b = unlist(lapply(share_lists, repb))
    
    tibble(Protein.A = long24a, Protein.B = long24b) %>% return() # Output pairs
  }) %>% bind_rows %>%
    group_by(Protein.A, Protein.B) %>%
    summarise(Co.Apex.Score = n() / num_reps) %>%
    ungroup # Sum and divide by number of reps
  
  return(coapex_scores)
}

###############START PHAT FUNCTION HERE
get_feature_df = function(raw, pairs = NULL, pearson = F, ncc = F, coapex = F, pids = seq(nrow(raw)), coapex_structure = NULL) {
  
  raw %<>% as.matrix
  rownames(raw) = NULL
  colnames(raw) = NULL
  
  #Generate all pairs from just name and index if not provided
  if (is.null(pairs)) {
    feature_df = combn(seq(nrow(raw)), m = 2) %>% t %>% as.data.frame
  } else {
    if (is.character(pairs[[1]])) {
      pairs %<>% as.matrix %>% match(pids) %>% matrix(ncol = 2) %>% na.omit
    }
    
    feature_df = as.data.frame(pairs)
  }
  
  colnames(feature_df) = c("Protein.A","Protein.B")

  #add euclidean distance for each pair
  x1 = raw[feature_df[[1]],]
  x2 = raw[feature_df[[2]],]
  
  feature_df$Euclidean.Distance = (x1 - x2)^2 %>% apply(1, sum) %>% sqrt
  
  #make correlation matrix and melt to get pearson correlation
  raw_na = raw
  raw_na[raw_na == 0] = NA
  
  get_shared = Vectorize(function(i, j) sum(raw[i,] != 0 | raw[j,] != 0))
  feature_df %<>% mutate(
    Weights = get_shared(Protein.A, Protein.B)
  )
  
  if (pearson) feature_df %<>% mutate(
    Pearson.Correlation = mapply(FUN = cor,
                                 x = lapply(Protein.A, function(i) raw_na[i,]),
                                 y = lapply(Protein.B, function(i) raw_na[i,]),
                                 MoreArgs = list(use = "pairwise.complete.obs"))
  ) 

  
  #attach weights from the other melted df to this one (genes are in same order)
  #melted_cormat$Weights = melted_share$Weight
    
  if (coapex) {
    #Get Co-apex score and add to the other df, then normalise from 0 to 1
    apexpairs = getapexpairs(raw, coapex_structure = coapex_structure)
    feature_df %<>% left_join(apexpairs, by = c("Protein.A", "Protein.B"))
    feature_df$Co.Apex.Score[is.na(feature_df$Co.Apex.Score)] = 0
    if(any(feature_df$Co.Apex.Score > 0))
      feature_df$Co.Apex.Score = feature_df$Co.Apex.Score / max(feature_df$Co.Apex.Score) 
  }
  
  if(ncc == T){
    feature_df$Normalised.Cross.Correlation = apply(as.matrix(feature_df[, 1:2]), 1, get_ncc, raw_mat = raw)
  }
  
  #stick them together
  # feature_df = left_join(feature_df, melted_cormat, by = c("Protein.A","Protein.B"))
  
  #calculate weighted euclidean distance - should this be / sqrt(feature$df_weights)?
  feature_df$Weighted.Euclidean.Distance = feature_df$Euclidean.Distance / feature_df$Weights
  
  #get rid of weights and euclidean distance columns
  
  select_columns = c("Protein.A", "Protein.B")
  if (pearson)  select_columns %<>% c("Pearson.Correlation")
  if (ncc)      select_columns %<>% c("Normalised.Cross.Correlation")
  if (coapex)   select_columns %<>% c("Co.Apex.Score")
                select_columns %<>% c("Weighted.Euclidean.Distance")
  
  feature_df = feature_df[, select_columns]

  #return protein names
  
  feature_df$Protein.A = pids[feature_df$Protein.A]
  feature_df$Protein.B = pids[feature_df$Protein.B]
  
  return(feature_df)
}

shuffle = function(x) {
  if (is.data.frame(x)) len = nrow(x) else len = length(x)
  neworder = sample(seq(len), size = len, replace = FALSE)
  if (is.data.frame(x)) return(x[neworder,]) else return(x[neworder])
}

generate_folds = function(pids, gold_standard, nfolds, seed = 1) {
  set.seed(seed)
  pids = pids[!grepl("^NA\\.\\.[0-9]*", pids)]
  colnames(gold_standard) = c("Protein_A", "Protein_B")
  
  gs_in_pids = data.frame(gold_standard$Protein_A %in% pids, gold_standard$Protein_B %in% pids) %>% apply(1, all)
  filtered_gs = gold_standard[gs_in_pids,] %>% shuffle
  
  split_groups = rep(seq(nfolds), length.out = nrow(filtered_gs))
  gs_groups = split(filtered_gs, split_groups)
  
  positive_proteins = c(gold_standard$Protein_A, gold_standard$Protein_B) %>% unique
  negative_proteins = pids[!pids %in% positive_proteins] %>%
    shuffle
  
  split_groups = rep(seq(nfolds), length.out = length(negative_proteins))
  neg_groups = negative_proteins %>% split(., split_groups)
  
  return(tibble(gs_groups = gs_groups,
              negative_proteins = neg_groups))
  
}

generate_interactions = function (gs, np, seed = 1) {
  set.seed(seed)

  # Negative links
  link_positive = c(gs[[1]], gs[[2]])
  link_negative = sample(np, size = length(link_positive), replace = TRUE)
  negative_links = matrix(c(link_positive, link_negative), ncol = 2) %>% apply(1, sample, size = 2, replace = FALSE) %>% t %>% as_tibble
  colnames(negative_links) = c("Protein_A", "Protein_B")
  
  positive_links = tibble(Protein.A = c(gs[[1]], gs[[2]]),
                          Protein.B = c(gs[[2]], gs[[1]]))
  colnames(positive_links) = c("Protein_A", "Protein_B")

  link_pids = bind_rows(positive_links, negative_links)
  link_labels = c(rep(1, nrow(positive_links)), rep(0, nrow(negative_links)))
  
  return(list(pids = link_pids, labels = link_labels))
}

generate_featureset = function(pairs, dat, pids, seed = 1, impute0 = FALSE, include_raw = FALSE, pearson = FALSE, ncc = FALSE, coapex = FALSE, coapex_structure = NULL) {
  set.seed(seed)
  
  out = list()
  
  shallow_features = get_feature_df(
    pairs, raw = dat,
    pearson = pearson,
    ncc = ncc,
    coapex = coapex,
    pids = pids,
    coapex_structure = coapex_structure
  )
  link_idx = pairs %>% as.matrix %>% match(pids) %>% matrix(ncol = 2)
  
  # Imputation
  if (impute0) {
    dat %<>% mutate_all(na_if, 0) %>% as.matrix %>%
      impute.knn(colmax = 0.99) %>%
      extract2("data") %>% as.data.frame
  }
  
  # Bind raw data rows together according to pair protein IDs
  raw_features = bind_cols(dat[link_idx[,1],], dat[link_idx[,2],])
  colnames(raw_features) = paste0("raw_", rep(c("A", "B"), each = ncol(dat)), "_", seq(ncol(dat)))
  
  shallow_features %<>% dplyr::select(-1, -2)
  out = bind_cols(raw_features, shallow_features)
  feature_colnames = colnames(out)
  out %<>% as.matrix %>% impute.knn(colmax = 0.99) %>% extract2("data") %>% as_tibble
  colnames(out) = feature_colnames
  
  if (!include_raw) out = out[,!grepl("raw_", feature_colnames)]
  
  return(out)
}

