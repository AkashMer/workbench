# Manual K-fold CV for gamlss models with re() random effects
library(here)
library(tidyverse)
library(tidymodels)
library(gamlss)

# Pin the here() root
here::i_am("scripts/model-cv.R")
data_path <- here("data")

# Load data
dataconf <- read_csv(file.path(data_path, "dataconf.csv"), show_col_types = FALSE)

# Prepare the data for train and test splits
data <- dataconf %>%
  # Convert the subject column into a factor
  mutate(subj = factor(as.integer(str_remove(subj, "^s")))) %>%
  # Convert the condition column into a factor
  mutate(condition = factor(condition, levels = c("noFB", "FB")))

# Split into train (80%) and test (20%) by whole subjects
set.seed(246) # For reproducibility
data_split <- group_initial_split(data, group = subj, prop = 0.8)
train <- training(data_split)

# Define the number of cycles to run for modelling and non-verbose output
gamlss_control <- gamlss.control(n.cyc = 300, trace = FALSE)

# Assign each unique subject to one of k folds
assign_subj_folds <- function(data, subj_col, k, seed) {
  set.seed(seed) # For reproducibility
  # One row per subject, randomly assigned a fold number
  subj_fold <- tibble(subj = unique(data[[subj_col]])) %>%
    mutate(fold = sample(rep_len(seq_len(k), n())))
  names(subj_fold)[1] <- subj_col
  # Expand subject-level folds back out to every trial row
  data %>% left_join(subj_fold, by = subj_col)
}

# Shared log file for all models' CV runs
cv_log_file <- file.path(data_path, "cv-log.txt")

# -- Unisensory Model --
if (FALSE) {
# Internal representation of the rotation, mu fixed since it is known not to vary
unisensory_mu_candidates <- list(
  intercept = error ~ 1 + re(random = ~1|subj, level = 0)
)

# Candidate sigma and nu formulas for target and trial_duration
unisensory_param_candidates <- list(
  target = ~ target + re(random = ~1|subj, level = 0),
  trial_duration = ~ trial_duration + re(random = ~1|subj, level = 0),
  additive = ~ target + trial_duration + re(random = ~1|subj, level = 0),
  interaction = ~ target * trial_duration + re(random = ~1|subj, level = 0)
)

# Assign whole subjects to folds for the unisensory CV run
unisensory_k <- 5
unisensory_folded <- assign_subj_folds(train, "subj", unisensory_k, seed = 246)

# Every combination of sigma and nu candidates to compare
unisensory_grid <- expand_grid(
  sigma_name = names(unisensory_param_candidates),
  nu_name = names(unisensory_param_candidates)
)
unisensory_nll <- numeric(nrow(unisensory_grid))

# Blank line then model name for log
cat("\n", "unisensory", "\n", file = cv_log_file, append = TRUE)

# Fit and evaluate each sigma-nu combination in turn
for (row in seq_len(nrow(unisensory_grid))) {
  sigma_form <- unisensory_param_candidates[[unisensory_grid$sigma_name[row]]]
  nu_form <- unisensory_param_candidates[[unisensory_grid$nu_name[row]]]
  fold_nll <- numeric(unisensory_k)
  # Refit on k-1 folds, evaluate on the held-out fold, once per fold
  for (fold_id in seq_len(unisensory_k)) {
    fit_data <- unisensory_folded %>% filter(fold != fold_id)
    held_data <- unisensory_folded %>% filter(fold == fold_id)
    # Skip a fold if gamlss()/predict() errors
    fold_nll[fold_id] <- tryCatch({
      # Fit gamlss() directly rather than through gamlssCV, since re() breaks inside gamlssCV/gamlssVGD
      m <- gamlss(
        formula = unisensory_mu_candidates$intercept,
        sigma.formula = sigma_form,
        nu.formula = nu_form,
        family = SN2(),
        data = fit_data,
        control = gamlss_control
      )
      # Predict mu, sigma and nu on the held-out fold's subjects
      pred <- held_data %>%
        mutate(pred_mu = predict(m, what = "mu", newdata = ., type = "response", data = fit_data),
               pred_sigma = predict(m, what = "sigma", newdata = ., type = "response", data = fit_data),
               pred_nu = predict(m, what = "nu", newdata = ., type = "response", data = fit_data))
      # Held-out negative log-likelihood for this fold
      -sum(dSN2(pred$error, mu = pred$pred_mu, sigma = pred$pred_sigma, nu = pred$pred_nu, log = TRUE))
    }, error = function(e) NA_real_)
  }
  # Total held-out NLL across all folds, NA if any fold failed to fit
  unisensory_nll[row] <- sum(fold_nll)
  # Log progress for this row to the log file
  cat("row", row, "of", nrow(unisensory_grid), "done, nll =", unisensory_nll[row], "\n", file = cv_log_file, append = TRUE)
}

# Save results as derived data
unisensory_cv_results <- unisensory_grid %>% mutate(nll = unisensory_nll)
saveRDS(unisensory_cv_results, here("data", "cv-unisensory.rds"))
}

# -- Naive Multisensory Model --
if (FALSE) {
# Candidate mu formulas, whether location shifts by presence of visual feedback
naive_multisensory_mu_candidates <- list(
  intercept = error ~ 1 + re(random = ~1|subj, level = 0),
  condition = error ~ condition + re(random = ~1|subj, level = 0)
)

# Candidate sigma and nu formulas, built on the winning unisensory formula (target + trial_duration) plus condition
naive_multisensory_param_candidates <- list(
  additive = ~ target + trial_duration + condition + re(random = ~1|subj, level = 0),
  interaction = ~ (target + trial_duration) * condition + re(random = ~1|subj, level = 0)
)

# Assign whole subjects to folds for the naive multisensory CV run
naive_multisensory_k <- 5
naive_multisensory_folded <- assign_subj_folds(train, "subj", naive_multisensory_k, seed = 246)

# Every combination of mu, sigma and nu candidates to compare
naive_multisensory_grid <- expand_grid(
  mu_name = names(naive_multisensory_mu_candidates),
  sigma_name = names(naive_multisensory_param_candidates),
  nu_name = names(naive_multisensory_param_candidates)
)
naive_multisensory_nll <- numeric(nrow(naive_multisensory_grid))

# Blank line then model name for log
cat("\n", "naive_multisensory", "\n", file = cv_log_file, append = TRUE)

# Fit and evaluate each mu-sigma-nu combination in turn
for (row in seq_len(nrow(naive_multisensory_grid))) {
  mu_form <- naive_multisensory_mu_candidates[[naive_multisensory_grid$mu_name[row]]]
  sigma_form <- naive_multisensory_param_candidates[[naive_multisensory_grid$sigma_name[row]]]
  nu_form <- naive_multisensory_param_candidates[[naive_multisensory_grid$nu_name[row]]]
  fold_nll <- numeric(naive_multisensory_k)
  # Refit on k-1 folds, evaluate on the held-out fold, once per fold
  for (fold_id in seq_len(naive_multisensory_k)) {
    fit_data <- naive_multisensory_folded %>% filter(fold != fold_id)
    held_data <- naive_multisensory_folded %>% filter(fold == fold_id)
    # Skip a fold if gamlss()/predict() errors
    fold_nll[fold_id] <- tryCatch({
      # Fit gamlss() directly rather than through gamlssCV, since re() breaks inside gamlssCV/gamlssVGD
      m <- gamlss(
        formula = mu_form,
        sigma.formula = sigma_form,
        nu.formula = nu_form,
        family = SN2(),
        data = fit_data,
        control = gamlss_control
      )
      # Predict mu, sigma and nu on the held-out fold's subjects
      pred <- held_data %>%
        mutate(pred_mu = predict(m, what = "mu", newdata = ., type = "response", data = fit_data),
               pred_sigma = predict(m, what = "sigma", newdata = ., type = "response", data = fit_data),
               pred_nu = predict(m, what = "nu", newdata = ., type = "response", data = fit_data))
      # Held-out negative log-likelihood for this fold
      -sum(dSN2(pred$error, mu = pred$pred_mu, sigma = pred$pred_sigma, nu = pred$pred_nu, log = TRUE))
    }, error = function(e) NA_real_)
  }
  # Total held-out NLL across all folds, NA if any fold failed to fit
  naive_multisensory_nll[row] <- sum(fold_nll)
  # Log progress for this row to the log file
  cat("row", row, "of", nrow(naive_multisensory_grid), "done, nll =", naive_multisensory_nll[row], "\n", file = cv_log_file, append = TRUE)
}

# Save results as derived data
naive_multisensory_cv_results <- naive_multisensory_grid %>% mutate(nll = naive_multisensory_nll)
saveRDS(naive_multisensory_cv_results, here("data", "cv-naive-multisensory.rds"))
}

# -- Multisensory Model --
if (FALSE) {
# Candidate mu formulas, whether location shifts by signed feedback offset
multisensory_mu_candidates <- list(
  intercept = error ~ 1 + re(random = ~1|subj, level = 0),
  fb_offset = error ~ fb_offset + re(random = ~1|subj, level = 0)
)

# Candidate sigma and nu formulas, built on the winning unisensory formula (target + trial_duration) plus fb_offset and fb_time
multisensory_param_candidates <- list(
  additive = ~ target + trial_duration + fb_offset + re(random = ~1|subj, level = 0),
  base_offset_interaction = ~ (target + trial_duration) * fb_offset + re(random = ~1|subj, level = 0),
  additive_fb_time = ~ target + trial_duration + fb_offset + fb_time + re(random = ~1|subj, level = 0),
  offset_fb_time_interaction = ~ target + trial_duration + fb_offset * fb_time + re(random = ~1|subj, level = 0),
  base_offset_interaction_fb_time = ~ (target + trial_duration) * fb_offset + fb_time + re(random = ~1|subj, level = 0)
)

# Assign whole subjects to folds for the multisensory CV run
multisensory_k <- 5
multisensory_folded <- assign_subj_folds(train, "subj", multisensory_k, seed = 246)

# Every combination of mu, sigma and nu candidates to compare
multisensory_grid <- expand_grid(
  mu_name = names(multisensory_mu_candidates),
  sigma_name = names(multisensory_param_candidates),
  nu_name = names(multisensory_param_candidates)
)
multisensory_nll <- numeric(nrow(multisensory_grid))

# Blank line then model name for log
cat("\n", "multisensory", "\n", file = cv_log_file, append = TRUE)

# Fit and evaluate each mu-sigma-nu combination in turn
for (row in seq_len(nrow(multisensory_grid))) {
  mu_form <- multisensory_mu_candidates[[multisensory_grid$mu_name[row]]]
  sigma_form <- multisensory_param_candidates[[multisensory_grid$sigma_name[row]]]
  nu_form <- multisensory_param_candidates[[multisensory_grid$nu_name[row]]]
  fold_nll <- numeric(multisensory_k)
  # Refit on k-1 folds, evaluate on the held-out fold, once per fold
  for (fold_id in seq_len(multisensory_k)) {
    fit_data <- multisensory_folded %>% filter(fold != fold_id)
    held_data <- multisensory_folded %>% filter(fold == fold_id)
    # Skip a fold if gamlss()/predict() errors
    fold_nll[fold_id] <- tryCatch({
      # Fit gamlss() directly rather than through gamlssCV, since re() breaks inside gamlssCV/gamlssVGD
      m <- gamlss(
        formula = mu_form,
        sigma.formula = sigma_form,
        nu.formula = nu_form,
        family = SN2(),
        data = fit_data,
        control = gamlss_control
      )
      # Predict mu, sigma and nu on the held-out fold's subjects
      pred <- held_data %>%
        mutate(pred_mu = predict(m, what = "mu", newdata = ., type = "response", data = fit_data),
               pred_sigma = predict(m, what = "sigma", newdata = ., type = "response", data = fit_data),
               pred_nu = predict(m, what = "nu", newdata = ., type = "response", data = fit_data))
      # Held-out negative log-likelihood for this fold
      -sum(dSN2(pred$error, mu = pred$pred_mu, sigma = pred$pred_sigma, nu = pred$pred_nu, log = TRUE))
    }, error = function(e) NA_real_)
  }
  # Total held-out NLL across all folds, NA if any fold failed to fit
  multisensory_nll[row] <- sum(fold_nll)
  # Log progress for this row to the log file
  cat("row", row, "of", nrow(multisensory_grid), "done, nll =", multisensory_nll[row], "\n", file = cv_log_file, append = TRUE)
}

# Save results as derived data
multisensory_cv_results <- multisensory_grid %>% mutate(nll = multisensory_nll)
saveRDS(multisensory_cv_results, here("data", "cv-multisensory.rds"))
}

# -- Multisensory Model, isolating target/trial_duration/fb_time interactions with fb_offset --
multisensory2_mu_formula <- error ~ 1 + re(random = ~1|subj, level = 0)

# The 2 confirmed winners from the first multisensory search
multisensory2_winners <- list(
  base_offset_interaction = ~ (target + trial_duration) * fb_offset + re(random = ~1|subj, level = 0),
  base_offset_interaction_fb_time = ~ (target + trial_duration) * fb_offset + fb_time + re(random = ~1|subj, level = 0)
)

# New candidates isolating which base term interacts with fb_offset, and whether fb_offset:fb_time adds anything
multisensory2_new_candidates <- list(
  target_offset_interaction = ~ target + trial_duration + fb_offset + target:fb_offset + re(random = ~1|subj, level = 0),
  duration_offset_interaction = ~ target + trial_duration + fb_offset + trial_duration:fb_offset + re(random = ~1|subj, level = 0),
  target_offset_interaction_fb_time = ~ target + trial_duration + fb_offset + target:fb_offset + fb_time + re(random = ~1|subj, level = 0),
  duration_offset_interaction_fb_time = ~ target + trial_duration + fb_offset + trial_duration:fb_offset + fb_time + re(random = ~1|subj, level = 0),
  base_offset_interaction_offset_fb_time_interaction = ~ target + trial_duration + fb_offset + target:fb_offset + trial_duration:fb_offset + fb_offset:fb_time + re(random = ~1|subj, level = 0),
  target_offset_interaction_offset_fb_time_interaction = ~ target + trial_duration + fb_offset + target:fb_offset + fb_offset:fb_time + re(random = ~1|subj, level = 0),
  duration_offset_interaction_offset_fb_time_interaction = ~ target + trial_duration + fb_offset + trial_duration:fb_offset + fb_offset:fb_time + re(random = ~1|subj, level = 0)
)

# Assign whole subjects to folds for this follow-up CV run
multisensory2_k <- 5
multisensory2_folded <- assign_subj_folds(train, "subj", multisensory2_k, seed = 246)

# Combined lookup of winners and new candidates
multisensory2_param_candidates <- c(multisensory2_winners, multisensory2_new_candidates)

# Test each new sigma candidate against the confirmed nu winner, and each new nu candidate against the confirmed sigma winner
multisensory2_grid <- bind_rows(
  tibble(sigma_name = names(multisensory2_new_candidates), nu_name = "base_offset_interaction_fb_time"),
  tibble(sigma_name = "base_offset_interaction", nu_name = names(multisensory2_new_candidates))
)
multisensory2_nll <- numeric(nrow(multisensory2_grid))

# Blank line then model name for log
cat("\n", "multisensory_interactions", "\n", file = cv_log_file, append = TRUE)

# Fit and evaluate each sigma-nu combination in turn
for (row in seq_len(nrow(multisensory2_grid))) {
  sigma_form <- multisensory2_param_candidates[[multisensory2_grid$sigma_name[row]]]
  nu_form <- multisensory2_param_candidates[[multisensory2_grid$nu_name[row]]]
  fold_nll <- numeric(multisensory2_k)
  # Refit on k-1 folds, evaluate on the held-out fold, once per fold
  for (fold_id in seq_len(multisensory2_k)) {
    fit_data <- multisensory2_folded %>% filter(fold != fold_id)
    held_data <- multisensory2_folded %>% filter(fold == fold_id)
    # Skip a fold if gamlss()/predict() errors
    fold_nll[fold_id] <- tryCatch({
      # Fit gamlss() directly rather than through gamlssCV, since re() breaks inside gamlssCV/gamlssVGD
      m <- gamlss(
        formula = multisensory2_mu_formula,
        sigma.formula = sigma_form,
        nu.formula = nu_form,
        family = SN2(),
        data = fit_data,
        control = gamlss_control
      )
      # Predict mu, sigma and nu on the held-out fold's subjects
      pred <- held_data %>%
        mutate(pred_mu = predict(m, what = "mu", newdata = ., type = "response", data = fit_data),
               pred_sigma = predict(m, what = "sigma", newdata = ., type = "response", data = fit_data),
               pred_nu = predict(m, what = "nu", newdata = ., type = "response", data = fit_data))
      # Held-out negative log-likelihood for this fold
      -sum(dSN2(pred$error, mu = pred$pred_mu, sigma = pred$pred_sigma, nu = pred$pred_nu, log = TRUE))
    }, error = function(e) NA_real_)
  }
  # Total held-out NLL across all folds, NA if any fold failed to fit
  multisensory2_nll[row] <- sum(fold_nll)
  # Log progress for this row to the log file
  cat("row", row, "of", nrow(multisensory2_grid), "done, nll =", multisensory2_nll[row], "\n", file = cv_log_file, append = TRUE)
}

# Save results as derived data
multisensory2_cv_results <- multisensory2_grid %>% mutate(nll = multisensory2_nll)
saveRDS(multisensory2_cv_results, here("data", "cv-multisensory-interactions.rds"))
