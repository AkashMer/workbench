# Load necessary libraries
library(here)
library(tidyverse)
library(gamlss)

# Pin the here() root
here::i_am("scripts/model-cv.R")
data_path <- here("data")

# Load data
dataconf <- read_csv(file.path(data_path, "dataconf.csv"), show_col_types = FALSE)
# Convert subject and condition columns into factors
data <- dataconf %>%
  mutate(subj = factor(as.integer(str_remove(subj, "^s")))) %>%
  mutate(condition = factor(condition, levels = c("noFB", "FB")))
# Classical, smooth and multisensory only use FB trials
fb_data <- data %>% filter(condition == "FB")

# Define the number of folds
k <- 5
seed <- 246 # For reproducibility

# Assign each unique subject to one of k folds, expand back out to trial rows
assign_subj_folds <- function(d, k, seed) {
  set.seed(seed)
  subj_fold <- tibble(subj = unique(d$subj)) %>%
    mutate(fold = sample(rep_len(seq_len(k), n())))
  d %>% left_join(subj_fold, by = "subj")
}

# Fold assignment on FB trials, unisensory's split derived from this below
folded <- assign_subj_folds(fb_data, k, seed)

# Save logs
log_file <- file.path(data_path, "cv-log.txt")
cat("CV run started", format(Sys.time()), "\n", file = log_file)

# Define model formulas
unisensory_mu_formula <- error ~ pbc(target) + pb(trial_duration) + re(random = ~1|subj, level = 0)
unisensory_sigma_formula <- ~ pbc(target) + pb(trial_duration) + re(random = ~1|subj, level = 0)

visual_only_mu_formula <- error ~ pbc(target) + pb(fb_offset) + re(random = ~1|subj, level = 0)
visual_only_sigma_formula <- ~ pbc(target) + pb(fb_offset) + re(random = ~1|subj, level = 0)

multisensory_mu_formula <- error ~ pbc(target) + pb(trial_duration) + pb(fb_offset) + pb(viewAmount) +
                                    pvc(target, by = fb_offset) +
                                    pvc(trial_duration, by = fb_offset) +
                                    pvc(viewAmount, by = fb_offset) +
                                    re(random = ~1|subj, level = 0)
multisensory_sigma_formula <- ~ pbc(target) + pb(trial_duration) + pb(fb_offset) + pb(viewAmount) +
                                pvc(target, by = fb_offset) +
                                pvc(trial_duration, by = fb_offset) +
                                pvc(viewAmount, by = fb_offset) +
                                re(random = ~1|subj, level = 0)

# Define the distribution family for error
family <- SST()

# Precision-weighted Bayesian-optimal combination of unisensory + visual-only predicted mu/sigma
classical_predict <- function(uni_mu, uni_sigma, vis_mu, vis_sigma) {
  uni_prec <- 1 / uni_sigma^2
  vis_prec <- 1 / vis_sigma^2
  (uni_mu * uni_prec + vis_mu * vis_prec) / (uni_prec + vis_prec)
}

# Compute RMSE for predictions vs actual
rmse <- function(pred, actual) sqrt(mean((pred - actual)^2, na.rm = TRUE))

# Kording et al. (2007) posterior probability of common cause, from the two cues' fitted mu/sigma
# fb_offset stands in as the observed conflict between the two cues
posterior_common_cause <- function(uni_mu, uni_sigma, vis_mu, vis_sigma, fb_offset, p_common) {
  var_sum <- uni_sigma^2 + vis_sigma^2
  lik_common <- dnorm(fb_offset, mean = 0, sd = sqrt(var_sum))
  lik_indep <- dnorm(fb_offset, mean = 0, sd = sqrt(var_sum + 1e4)) # broad, ~uninformative prior on independent causes
  num <- lik_common * p_common
  denom <- num + lik_indep * (1 - p_common)
  num / denom
}

# Model-averaged estimate: p(C=1) * combined (classical) + p(C=2) * unisensory alone (segregated)
smooth_predict <- function(uni_mu, uni_sigma, vis_mu, vis_sigma, fb_offset, p_common) {
  p_common_post <- posterior_common_cause(uni_mu, uni_sigma, vis_mu, vis_sigma, fb_offset, p_common)
  combined <- classical_predict(uni_mu, uni_sigma, vis_mu, vis_sigma)
  p_common_post * combined + (1 - p_common_post) * uni_mu
}

# Grid search p_common on training predictions, minimizing training RMSE
fit_p_common <- function(uni_mu, uni_sigma, vis_mu, vis_sigma, fb_offset, actual) {
  grid <- seq(0.1, 0.9, by = 0.05)
  train_rmse <- map_dbl(grid, function(p) {
    pred <- smooth_predict(uni_mu, uni_sigma, vis_mu, vis_sigma, fb_offset, p)
    rmse(pred, actual)
  })
  grid[which.min(train_rmse)]
}
# Initialize results table
results <- tibble()

# Loop through each main and leave out fold pair
for (fold_id in seq_len(k)) {
  cat("fold", fold_id, "start", format(Sys.time()), "\n", file = log_file, append = TRUE)

  # FB-trial fit/held split for classical, smooth and multisensory
  fit_data <- folded %>% filter(fold != fold_id)
  held_data <- folded %>% filter(fold == fold_id)

  # Unisensory's fit/held split: same held-out subjects, but full data (noFB + FB)
  fit_data_full <- data %>%
    semi_join(fit_data %>% distinct(subj), by = "subj") %>%
    anti_join(held_data %>% select(subj, trial), by = c("subj", "trial"))
  held_data_full <- data %>%
    semi_join(held_data %>% distinct(subj), by = "subj") %>%
    inner_join(held_data %>% select(subj, trial), by = c("subj", "trial"))

  # Fit unisensory on this fold's training subjects, full data
  uni_fit <- tryCatch(
    gamlss(
      formula = unisensory_mu_formula, sigma.formula = unisensory_sigma_formula,
      family = family, data = fit_data_full, method = mixed(20, 100),
      control = gamlss.control(n.cyc = 300, trace = FALSE)
    ),
    error = function(e) {
      cat("fold", fold_id, "unisensory failed:", conditionMessage(e), "\n", file = log_file, append = TRUE)
      NULL
    }
  )

  # Fit visual-only on this fold's training FB trials
  vis_fit <- tryCatch(
    gamlss(
      formula = visual_only_mu_formula, sigma.formula = visual_only_sigma_formula,
      family = family, data = fit_data, method = mixed(20, 100),
      control = gamlss.control(n.cyc = 300, trace = FALSE)
    ),
    error = function(e) {
      cat("fold", fold_id, "visual_only failed:", conditionMessage(e), "\n", file = log_file, append = TRUE)
      NULL
    }
  )

  # Fit multisensory on this fold's training FB trials
  multi_fit <- tryCatch(
    gamlss(
      formula = multisensory_mu_formula, sigma.formula = multisensory_sigma_formula,
      family = family, data = fit_data, method = mixed(20, 100),
      control = gamlss.control(n.cyc = 300, trace = FALSE)
    ),
    error = function(e) {
      cat("fold", fold_id, "multisensory failed:", conditionMessage(e), "\n", file = log_file, append = TRUE)
      NULL
    }
  )

  # NA until filled in below, only for fits that succeeded
  fold_rmse <- tibble(fold = fold_id, model = c("unisensory", "classical", "smooth", "multisensory"), rmse = NA_real_)

  # Unisensory RMSE on held-out full data, predictAll for correct multi-parameter prediction
  if (!is.null(uni_fit)) {
    uni_pred_held <- predictAll(uni_fit, newdata = held_data_full, type = "response", data = fit_data_full, output = "list")
    fold_rmse$rmse[fold_rmse$model == "unisensory"] <- rmse(uni_pred_held$mu, held_data_full$error)
  }

  # Classical + smooth RMSE, combine unisensory + visual-only predictions on held-out FB trials
  if (!is.null(uni_fit) && !is.null(vis_fit)) {
    # uni_fit trained without the fold column, drop it here so newdata columns match
    uni_pred_held <- predictAll(uni_fit, newdata = held_data %>% select(-fold), type = "response", data = fit_data_full, output = "list")
    vis_pred_held <- predictAll(vis_fit, newdata = held_data, type = "response", data = fit_data, output = "list")

    classical_pred <- classical_predict(uni_pred_held$mu, uni_pred_held$sigma, vis_pred_held$mu, vis_pred_held$sigma)
    fold_rmse$rmse[fold_rmse$model == "classical"] <- rmse(classical_pred, held_data$error)

    # p_common grid-searched on this fold's training predictions
    uni_pred_fit <- predictAll(uni_fit, newdata = fit_data %>% select(-fold), type = "response", data = fit_data_full, output = "list")
    vis_pred_fit <- predictAll(vis_fit, newdata = fit_data, type = "response", data = fit_data, output = "list")

    p_common <- fit_p_common(
      uni_pred_fit$mu, uni_pred_fit$sigma, vis_pred_fit$mu, vis_pred_fit$sigma, fit_data$fb_offset, fit_data$error
    )
    cat("fold", fold_id, "p_common", p_common, "\n", file = log_file, append = TRUE)

    smooth_pred <- smooth_predict(
      uni_pred_held$mu, uni_pred_held$sigma, vis_pred_held$mu, vis_pred_held$sigma, held_data$fb_offset, p_common
    )
    fold_rmse$rmse[fold_rmse$model == "smooth"] <- rmse(smooth_pred, held_data$error)
  }

  # Multisensory RMSE on held-out FB trials
  if (!is.null(multi_fit)) {
    multi_pred_held <- predictAll(multi_fit, newdata = held_data, type = "response", data = fit_data, output = "list")
    fold_rmse$rmse[fold_rmse$model == "multisensory"] <- rmse(multi_pred_held$mu, held_data$error)
  }

  results <- bind_rows(results, fold_rmse)
  cat("fold", fold_id, "done", format(Sys.time()), "\n", file = log_file, append = TRUE)
}

# Save results as R object
saveRDS(results, file.path(data_path, "cv-results.rds"))
cat("CV run finished", format(Sys.time()), "\n", file = log_file, append = TRUE)
