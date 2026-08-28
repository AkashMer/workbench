# Initialize required libraries
library(tidyverse)
library(here)
library(circular)
library(rmcorr)
library(boot)
library(quantreg)

# Download the data files from the github repo
# 1. Get the data path
data_path <- here("data", "heading-estimation-multisensory-proc")
# 2. Make the directory if it does not exist already
dir.create(data_path, recursive = TRUE, showWarnings = FALSE)
# 3. Download the data file to data_path if not already present
if (!file.exists(file.path(data_path, "data.csv"))) {
  data_url <- "https://raw.githubusercontent.com/sharootonian/CombinationAndCompetitionHeadingDirection/main/data/data.csv"
  download.file(data_url, file.path(data_path, "data.csv"), mode = "wb")
}
if (!file.exists(file.path(data_path, "dataconf.csv"))) {
  dataconf_url <- "https://raw.githubusercontent.com/sharootonian/CombinationAndCompetitionHeadingDirection/main/data/dataconf.csv"
  download.file(dataconf_url, file.path(data_path, "dataconf.csv"), mode = "wb")
}

# Load the file and check the structure
bva_data <- read_csv(file.path(data_path, "data.csv"))
colnames(bva_data)
dim(bva_data)

# Number of unique participants
length(unique(bva_data$subj))
# 30 participants, confirmed all participant data present

# Get a general description of the dataframe
summary(bva_data)
# Max 399 trials per participant
# target: seems uniformly distribution from 0 - 360
# respond makes sense to be > 360 but then error is between -180 to 180: needs further investigation
# trial_duration: most likely represents the duration of whole trial, no way to get response time (needs confirmation)
# I wanna confirm if fb and fb_true match and when offset is zero
# fb_time: Most likely time from start to onset of feedback image

# Load and get the description of dataconf file as well
bva_dataconf <- read_csv(file.path(data_path, "dataconf.csv"))
summary(bva_dataconf)
# Only two new columns
# conf: A post-hoc cone angle asked from the participant for the door position in their response view
# viewAmount: needs confirmation but the view angle traversed while feedback was visible
# ie. serves as a measure of how much attention was payed to the feedback

# How does high respond correspond to error values?
bva_data %>% filter(respond > 500)
# error = respond - target
summary(bva_data$respond - bva_data$target - bva_data$error)
# Confirmed => respond is the cumulative angle for the whole trial duration
# Hence, makes sense why > 360

# Confirm trial duration is the whole trial
ggplot(bva_data, aes(x = trial_duration)) +
  geom_histogram() # right-skewed in msec
# Is fb_time < trial_duration always?
mean(bva_data$fb_time < bva_data$trial_duration) # not 1
# Check which fb_time > trial_duration
bva_data %>% filter(fb_time > trial_duration) # zero rows
# Maybe equal?
bva_data %>% filter(fb_time == trial_duration)
# trial duration is start of encoding period to response button press

# Do fb and fb_true match when offset is zero
bva_data %>%
  mutate(fb_diff = fb - fb_true) %>%
  filter(condition == "FB" & fb_offset == 0) %>%
  summary()
# fb_offset is never zero
# On further confirmation, fb_offset is drawn two ways:
# 1. Gaussian with 0 mean and SD: 30 degrees (70% trials/participant)
# 2. Uniformly from -180 to 180 (30% trials/participant)

# Confirm viewAmount is zero for noFB trials
bva_dataconf %>%
  filter(condition == "noFB") %>%
  pull(viewAmount) %>%
  summary()
# All zeros, confirmed

# Create a new dataframe for EDA
eda_data <- bva_dataconf %>%
  # Convert the subject column into a factor
  mutate(subj = factor(as.integer(str_remove(subj, "^s")))) %>%
  # Convert the condition column into a factor
  mutate(condition = factor(condition, levels = c("noFB", "FB"))) %>%
  # Since feedback validity labels are not present, I will use label reliable for gaussian (0, 30) +/- 3SD
  mutate(fb_validity = factor(case_when(condition == "noFB" ~ NA,
                                        abs(fb_offset) <= 90 ~ "reliable",
                                        TRUE ~ "unreliable"),
                              levels = c("reliable", "unreliable"))) %>%
  # Create a new column to signify the three trial types
  mutate(trial_type = if_else(condition == "noFB", "noFB", as.character(fb_validity))) %>%
  mutate(trial_type = factor(trial_type, levels = c("noFB", "reliable", "unreliable")))

# Trial counts per subject for each trial type
bind_rows(
  eda_data %>% count(subj, group = condition) %>% mutate(facet = "condition"),
  eda_data %>% count(subj, group = trial_type) %>% mutate(facet = "trial_type")
) %>%
  group_by(facet, group) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup() %>%
  ggplot(aes(x = subj, y = prop, color = group, group = group)) +
  geom_point() +
  geom_line() +
  facet_wrap(~facet, ncol = 1)
# Some participants have low number of trials, but they are equally low for all trial types
# Like particiant s4, s5, s9, s10, s21(lowest)
# How are the included trials distributed across each subject?
eda_data %>%
  count(subj, trial) %>%
  ggplot(aes(x = trial, y = subj, fill = n)) +
  geom_tile()
# s4, s5, s7, s10: Truncated session
# s9, s14, s21 (highest), s24: Scattered dropped trials
# => Some trials might have been excluded from analysis
# Thus any learning behavior analysis will need to exclude the subjects with high scattered dropped trials

# Check how the sessions were structured
ggplot(eda_data, aes(x = trial, y = condition, color = condition)) +
  geom_point() +
  scale_color_brewer(palette = "Dark2")
# Each session had a FB block flanked by noFB blocks with 2 sessions in total (200 trials each)
# Now across trial types which included fb reliability
ggplot(eda_data, aes(x = trial, y = trial_type, color = trial_type)) +
  geom_point() +
  scale_color_brewer(palette = "Dark2")
# reliable and unreliable fb trials are randomly interspersed in the FB trial block

# Confirm the distribution of target
eda_data %>%
  # Divide target into 18 20 degree bins
  mutate(target_bin = cut(target, breaks = seq(0, 360, by = 20), include.lowest = TRUE)) %>%
  count(target_bin) %>%
  ggplot(aes(x = target_bin, y = n)) +
  geom_col(colour = "steelblue", fill = "steelblue") +
  # Change to a circular coordinate
  coord_polar(start = 0) +
  theme_bw()
# Fairly uniform distribution throughout, but lower number towards higher values

# Distribution of the respond variable
ggplot(eda_data, aes(x = respond)) +
  geom_histogram(binwidth = 20, color = "black", fill = "steelblue") +
  geom_vline(xintercept = mean(eda_data$respond), color = "#D55E00", linetype = "dashed") +
  geom_vline(xintercept = median(eda_data$respond), color = "#0072B2", linetype = "dashed") +
  annotate("text", x = mean(eda_data$respond), y = Inf, label = paste0("mean=", round(mean(eda_data$respond), 1)),
           color = "#D55E00", vjust = 2, hjust = -0.1) +
  annotate("text", x = median(eda_data$respond), y = Inf, label = paste0("median=", round(median(eda_data$respond), 1)),
           color = "#0072B2", vjust = 4, hjust = -0.1)
# Almost bell-shaped, with a lot of mass in the center and mean (193) === median (192.7)
# Check how the distribution differs for each trial type
ggplot(eda_data, aes(x = respond, fill = trial_type)) +
  geom_histogram(binwidth = 20, color = "black") +
  facet_wrap(~trial_type) +
  scale_fill_brewer(palette = "Dark2")
# Similar shape for all three matching the overall shape
# Check how respond varies with target bins
# Add the target bin columns (18 bins of 20 degrees)
eda_data <- eda_data %>%
  mutate(target_bin = cut(target, breaks = seq(0, 360, by = 20), include.lowest = TRUE))
# Plot against target bin overall
ggplot(eda_data, aes(x = target_bin, y = respond)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "target", y = "respond")
# Respond tracks target cleanly throughout, which makes sense since respond is cumulative
# The variance of respond grows with target => stimulus noise grows with higher target angles
# Confirm if the same relationship exists across conditions and trial types
# noFB vs FB
ggplot(eda_data, aes(x = target_bin, y = respond)) +
  geom_boxplot() +
  facet_wrap(~condition, ncol = 1) +
  theme(axis.text.x = element_blank()) +
  labs(x = "target", y = "respond")
# Similar pattern across conditions, but the uncertainity (variance) is higher in FB trials
# noFB vs reliable vs unreliable
ggplot(eda_data, aes(x = target_bin, y = respond)) +
  geom_boxplot() +
  facet_wrap(~trial_type, ncol = 1) +
  theme(axis.text.x = element_blank()) +
  labs(x = "target", y = "respond")
# Similar pattern across all trial types, but the uncertainity (variance) increases down
# the trial types => even unreliable feedback is not being discounted completely
# Key takeway: The noise in perception may need to be modeled by target, condition and fb_validity

# Distribution of the error variable
ggplot(eda_data, aes(x = error)) +
  geom_histogram(binwidth = 20, color = "black", fill = "steelblue") +
  geom_vline(xintercept = mean(eda_data$error), color = "#D55E00", linetype = "dashed") +
  geom_vline(xintercept = median(eda_data$error), color = "#0072B2", linetype = "dashed") +
  annotate("text", x = mean(eda_data$error), y = Inf, label = paste0("mean=", round(mean(eda_data$error), 1)),
           color = "#D55E00", vjust = 2, hjust = -0.1) +
  annotate("text", x = median(eda_data$error), y = Inf, label = paste0("median=", round(median(eda_data$error), 1)),
           color = "#0072B2", vjust = 4, hjust = -0.1)
# Right skewed with mean (18.1) > median (15.3): Why is error positively biased?
# Does the positive bias hold across trial types?
ggplot(eda_data, aes(x = error)) +
  geom_histogram(binwidth = 20, color = "black", fill = "steelblue") +
  geom_text(data = eda_data %>%
                      group_by(trial_type) %>%
                      summarise(mean_val = mean(error)),
              aes(x = mean_val, y = Inf, label = paste0("mean=", round(mean_val, 1))),
              color = "#D55E00", vjust = 2, hjust = -0.1) +
  geom_text(data = eda_data %>%
                      group_by(trial_type) %>%
                      summarise(median_val = median(error)),
              aes(x = median_val, y = Inf, label = paste0("median=", round(median_val, 1))),
              color = "#0072B2", vjust = 4, hjust = -0.1) +
  facet_wrap(~trial_type) +
  labs(x = "error") +
  scale_fill_brewer(palette = "Dark2")
# Similar shape across trial types as well, with reliable FB trials the most skewed (mean - median = 3.1 vs. ~1.8)
# Positive bias even noFB trials => behavioral tendency for overestimation
# Does the positive bias stay constant across target bins?
ggplot(eda_data, aes(x = target_bin, y = error)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "target", y = "error")
# Positive bias is maintained throughout, suggesting the bias is due to behavioral tendency for overestimation
# The variance grows with target => perception noise depends on target value
# Confirm if the same relationship exists across conditions and trial types
# noFB vs FB
ggplot(eda_data, aes(x = target_bin, y = error)) +
  geom_boxplot() +
  facet_wrap(~condition, ncol = 1) +
  theme(axis.text.x = element_blank()) +
  labs(x = "target", y = "error")
# Similar pattern holds across condition, but variance grows faster in FB trials
# noFB vs reliable vs unreliable
ggplot(eda_data, aes(x = target_bin, y = error)) +
  geom_boxplot() +
  facet_wrap(~trial_type, ncol = 1) +
  theme(axis.text.x = element_blank()) +
  labs(x = "target", y = "error")
# Positive bias trend breaks slightly for unreliable FB trials
# Variance grows fastest in unreliable FB trials, so the above deviation from trend could be a mixture of
# low trial count and the variance effect
# Key takeway: The noise in perception may need to be modeled by target, condition and fb_validity
# Is the positive bias for error driven by subject level data?
eda_data %>%
  group_by(subj) %>%
  summarise(mean_val = mean(error), median_val = median(error)) %>%
  mutate(gap = mean_val - median_val) %>%
  pivot_longer(cols = c(mean_val, median_val), names_to = "stat", values_to = "value") %>%
  ggplot(aes(x = subj, y = value, color = stat, group = stat)) +
  geom_point() +
  geom_line() +
  geom_text(data = . %>% distinct(subj, gap),
            aes(x = subj, y = Inf, label = round(gap, 1)),
            inherit.aes = FALSE, vjust = 1.5, size = 3, color = "black") +
  geom_hline(yintercept = 0) +
  scale_color_brewer(palette = "Dark2") +
  theme_bw()
# Majority of subjects have a positively biased error
# Exception: s2, s5, s21 (s5 and s21 have low trial count)
# Does the positive bias in error change over the course of a session?
ggplot(eda_data, aes(x = trial, y = error)) +
  geom_smooth()
# Positively biased across sessions as well with lowest error in the beginning of the 1st session

# Distribution of the trial_duration variable
ggplot(eda_data, aes(x = trial_duration / 1000)) +
  geom_histogram(color = "black", fill = "steelblue") +
  geom_vline(xintercept = mean(eda_data$trial_duration) / 1000, color = "#D55E00", linetype = "dashed") +
  geom_vline(xintercept = median(eda_data$trial_duration) / 1000, color = "#0072B2", linetype = "dashed") +
  annotate("text", x = mean(eda_data$trial_duration) / 1000, y = Inf,
           label = paste0("mean=", round(mean(eda_data$trial_duration) / 1000, 1)),
           color = "#D55E00", vjust = 2, hjust = -0.1) +
  annotate("text", x = median(eda_data$trial_duration) / 1000, y = Inf,
           label = paste0("median=", round(median(eda_data$trial_duration) / 1000, 1)),
           color = "#0072B2", vjust = 4, hjust = -0.1) +
  labs(x = "trial_duration (sec)")
# Right skewed with peak at 3-4 s, mean (4.5s) > median (4.1s) with a long tail
# Does the structure hold across trial types?
ggplot(eda_data, aes(x = trial_duration / 1000, fill = trial_type)) +
  geom_histogram(color = "black") +
  facet_wrap(~trial_type) +
  scale_fill_brewer(palette = "Dark2") +
  labs(x = "trial_duration (sec)")
# Similar structure holds across all trial types
# Hypothesis: Target angle should drive trial duration until a certain threshold,
# beyond which response time should dominate
# Does target angle drive trial duration?
ggplot(eda_data, aes(x = target_bin, y = trial_duration / 1000)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "target", y = "trial_duration (sec)")
# Plot view is being driven by the one outlier at 50 sec
# What is the structure of this outlier?
eda_data %>%
  filter(trial_duration / 1000 > 20) %>%
  arrange(desc(trial_duration))
# 8 in total: Subject 31 appears thrice; FB > noFB trials; majority of targets are > 120 (except 2)
# errors are varied; # unreliable = reliable trials
# Replot with the above 8 excluded
eda_data %>%
  filter(trial_duration / 1000 <= 20) %>%
  ggplot(aes(x = target_bin, y = trial_duration / 1000)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "target", y = "trial_duration (sec)")
# Monotonic increase across target bins; but the variance also increases (similar to respond/error)
# Due to lack of response time column; error can be used as a loose proxy for this time
# Check trial duration against error as a proxy for response time
eda_data %>%
  filter(trial_duration / 1000 <= 20) %>%
  ggplot(aes(x = error, y = trial_duration / 1000)) +
  geom_hex() +
  labs(x = "error", y = "trial_duration (sec)")
# No clear trend except low error trials had low trial duration
# => error may not be a good proxy for response time
# Another proxy for response time would be post fb response time = trial_duration - fb_time
# But this depends on trial_duration itself, so the difference may just track trial_duration
# Does the pattern hold across subjects?
eda_data %>%
  filter(trial_duration / 1000 <= 20) %>%
  ggplot(aes(x = subj, y = trial_duration / 1000)) +
  geom_boxplot() +
  labs(y = "trial_duration (sec)") +
  theme_bw()
# Fairly varied with some subjects having more longer trials than others
# Could the same subjects have higher variance across trials
# Trial duration across trials, per subject
eda_data %>%
  filter(trial_duration / 1000 <= 20) %>%
  ggplot(aes(x = trial, y = trial_duration / 1000)) +
  geom_line() +
  geom_smooth() +
  facet_wrap(~subj, ncol = 6) +
  labs(y = "trial_duration (sec)")
# Fairly similar across subjects, even those with more longer trials
# => Trial duration only shows a meaningful trend with variance which could
# be capturing the same behavioral effect as respond/error; the perception noise
# Lack of a formal response time measure => any RT analysis cannot be done

# feedback columns are controlled variables, thus no further EDA needed

# conf - subjective variable spread
# Distribution of the conf variable
ggplot(eda_data, aes(x = conf)) +
  geom_histogram(color = "black", fill = "steelblue") +
  geom_vline(xintercept = mean(eda_data$conf), color = "#D55E00", linetype = "dashed") +
  geom_vline(xintercept = median(eda_data$conf), color = "#0072B2", linetype = "dashed") +
  annotate("text", x = mean(eda_data$conf), y = Inf,
           label = paste0("mean=", round(mean(eda_data$conf), 1)),
           color = "#D55E00", vjust = 2, hjust = -0.1) +
  annotate("text", x = median(eda_data$conf), y = Inf,
           label = paste0("median=", round(median(eda_data$conf), 1)),
           color = "#0072B2", vjust = 4, hjust = -0.1)
# Most subjects were confident, but the higher uncertainity is varied
# thus producing a high mean median difference => consistent with a self-report variable
# A bump around 30-35 degrees
# Is this consistent across trial types?
ggplot(eda_data, aes(x = conf, fill = trial_type)) +
  geom_histogram(color = "black") +
  facet_wrap(~trial_type) +
  scale_fill_brewer(palette = "Dark2")
# Similar distribution across all 3 trial types
# But the bump around 30-35 degrees appears in all three
# Could the bump be subject driven?
ggplot(eda_data, aes(x = subj, y = conf)) +
  geom_boxplot() +
  theme_bw()
# Differing spreads across subjects => maybe coding temperaments
# s7, s10, s12, s18, s24 have IQR boxes in the 20-35 degree range unlike other subjects
# => subject driven bump in the overall distribution
# Does conf show a trial-order trend per subject?
ggplot(eda_data, aes(x = trial, y = conf)) +
  geom_line() +
  geom_smooth() +
  facet_wrap(~subj, ncol = 6)
# The temperament is consistent across trials for each subject
# Does subjective conf match actual error magnitude?
eda_data %>%
  mutate(abs_error_bin = cut(abs(error), breaks = seq(0, 180, by = 20), include.lowest = TRUE)) %>%
  ggplot(aes(x = abs_error_bin, y = conf)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "|error| (binned)", y = "conf")
# No trend, but higher outliers for lower error values; this is unexpected since
# a naive bayesian observer should be able to track the performance well
# Does the calibration pattern differ between noFB and FB trials?
eda_data %>%
  mutate(abs_error_bin = cut(abs(error), breaks = seq(0, 180, by = 20), include.lowest = TRUE)) %>%
  ggplot(aes(x = abs_error_bin, y = conf)) +
  geom_boxplot() +
  facet_wrap(~condition, ncol = 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "|error| (binned)", y = "conf")
# Similar pattern
# What about across trial_types?
eda_data %>%
  mutate(abs_error_bin = cut(abs(error), breaks = seq(0, 180, by = 20), include.lowest = TRUE)) %>%
  ggplot(aes(x = abs_error_bin, y = conf)) +
  geom_boxplot() +
  facet_wrap(~trial_type, ncol = 1) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "|error| (binned)", y = "conf")
# Similar pattern
# Does subjective conf relate to target ie task difficulty like respond/error?
ggplot(eda_data, aes(x = target_bin, y = conf)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "target", y = "conf")
# Increasing trend across target bins with increasing variance
# => subjective conf is being updated based on prior dependent on target
# Subjective conf depends on the target variable and not the objective error
# which makes sense since it is a subjective measure => rational bayesian observer

# Does viewAmount differ between reliable and unreliable feedback cues?
# ie. Is attention affected by the reliability of the feedback?
eda_data %>%
  filter(condition == "FB") %>%
  ggplot(aes(x = viewAmount, fill = trial_type)) +
  geom_histogram(color = "black") +
  facet_wrap(~trial_type) +
  scale_fill_brewer(palette = "Dark2")
# Similar structure across both trial types
# Does attention vary with magnitude of the fb_offset?
eda_data %>%
  filter(condition == "FB") %>%
  mutate(abs_offset_bin = cut(abs(fb_offset), breaks = seq(0, 180, by = 20), include.lowest = TRUE)) %>%
  ggplot(aes(x = abs_offset_bin, y = viewAmount)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "|fb_offset| (binned)", y = "viewAmount")
# Confirms the similar structure seen across trial types
# Does attention to the feedback cue (viewAmount) affect post-hoc confidence?
# Expectation: Lower viewAmount (higher attention) leads to lower conf (high confidence)
eda_data %>%
  filter(condition == "FB") %>%
  ggplot(aes(x = viewAmount, y = conf, color = trial_type)) +
  geom_point(alpha = 0.2) +
  geom_smooth() +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "viewAmount", y = "conf")
# No trend in the densest parts of the plot, across either trial types
# Does attention to the feedback cue (viewAmount) affect accuracy?
eda_data %>%
  filter(condition == "FB") %>%
  mutate(viewAmount_bin = cut(viewAmount, breaks = seq(0, 100, by = 20), include.lowest = TRUE)) %>%
  ggplot(aes(x = viewAmount_bin, y = abs(error))) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "viewAmount (binned)", y = "|error|")
# Accuracy is higher (low |error|) for trials with low viewAmount with increasing variance
# => abs(error) noise modelling should include viewAmount as well for FB trials

# Statistical Analysis
alpha <- 0.05
R_boot <- 1000

# 1. Is target actually uniformly distributed in a circle?
kuiper.test(circular(eda_data$target, units = "degrees"), alpha = alpha)
# t-stat = 2.8768; critical value = 1.747 => Reject H0 => non uniform
# Confirm across conditions
# noFB trials
eda_data %>%
  filter(condition == "noFB") %>%
  pull(target) %>%
  circular(units = "degrees") %>%
  kuiper.test(alpha = alpha)
# t-stat = 1.8274; critical value = 1.747 => Reject H0 => non uniform 
# FB trials
eda_data %>%
  filter(condition == "FB") %>%
  pull(target) %>%
  circular(units = "degrees") %>%
  kuiper.test(alpha = alpha)
# t-stat = 2.4396; critical value = 1.747 => Reject H0 => non uniform
# Confirm using mean direction and concentration of values
mean.circular(circular(eda_data$target, units = "degrees"))
rho.circular(circular(eda_data$target, units = "degrees"))
# Practically uniform, the power of the t-test was high due to so many trials
# Confirm if almost similar concentration of target angles is present across each subject
eda_data %>%
  summarise(circular_rho = rho.circular(circular(target, units = "degrees")), .by = subj) %>%
  mutate(is_outlier = abs(circular_rho - median(circular_rho)) > 2.5 * mad(circular_rho)) %>%
  filter(is_outlier)
# Only 3 subjects have larger concentration, but s7, s9 and s21 have known low trial count

# 2. Since respond's variance has same characteristics as error's variance,
#    Is respond's variance coming directly from error?
# ie. check if the last term in Var(respond) = var(target) + var(error) + 2 cov(target, error) -> 0
rmcorr(subj, target, error, dataset = eda_data) # repeated measure to avoid pooling across subjects
# cor = 0.03 with 95% conf interval [0.009 0.047]
# Negligeble => var(respond) is tracking var(error) => error can be used as the predicting variable

# 3. Is error variable truly positively biased?
wilcox.test(eda_data$error, mu = 0)
# p-value very small
# Confirm by finding the bootstrapped CI across subjects and trials
subjs <- unique(eda_data$subj)
median_error_boot <- boot(subjs, function(d, i) {
  samp <- map_dfr(d[i], ~ eda_data %>% filter(subj == .x))
  median(samp$error)
}, R = R_boot)
boot.ci(median_error_boot, type = "perc")
# [10.66 20.75] (95%)
# Statistically confirmed positive bias

# 4. Does error stays constant across 20 degree target bins?
# Controlling for individual subjects
error_slope_boot_by_bin <- boot(subjs, function(d, i) {
  samp <- map_dfr(d[i], ~ eda_data %>% filter(subj == .x))
  coef(rq(error ~ target, tau = 0.5, data = samp))["target"]
}, R = R_boot)
boot.ci(error_slope_boot_by_bin, type = "perc")
# Slope == 0; [-0.018 0.052]; confirms the EDA finding
# 5. Does the variance of error grow across 20 degree target bins?
# Controlling for individual subjects
error_sd_boot_by_bin <- eda_data %>%
  group_by(target_bin) %>%
  group_modify(~ {
    bin_data <- .x
    bin_subjs <- unique(bin_data$subj)
    b <- boot(bin_subjs, function(d, i) {
      samp <- map_dfr(d[i], ~ bin_data %>% filter(subj == .x))
      sd(samp$error)
    }, R = R_boot)
    ci <- boot.ci(b, type = "perc")$percent[4:5]
    tibble(sd = b$t0, ci_low = ci[1], ci_high = ci[2])
  })
ggplot(error_sd_boot_by_bin, aes(x = target_bin, y = sd)) +
  geom_pointrange(aes(ymin = ci_low, ymax = ci_high)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "target (binned)", y = "SD(error) with 95% bootstrap CI")
# Grows in two phases witch a plateau in the middle around 140 to 200 target angle
# But consecutive beans have 95% CI overlap
# => Variance of error ie. the perceptual uncerntainity does grow with target angle
# but the trend is not linear

# 6. Does error's median differ between noFB and FB conditions?
wilcox.test(error ~ condition, data = eda_data)
# Very low p-value
# Confirm using a 95% CI controlled for subjects
median_diff_boot <- boot(subjs, function(d, i) {
  samp <- map_dfr(d[i], ~ eda_data %>% filter(subj == .x))
  median(samp$error[samp$condition == "FB"]) - median(samp$error[samp$condition == "noFB"])
}, R = R_boot)
boot.ci(median_diff_boot, type = "perc")
# CI difference of [1.98 6.16]
# 7. Does error's variance differ between noFB and FB conditions?
# Controlled for subjects
condition_sd_boot <- eda_data %>%
  group_by(condition) %>%
  group_modify(~ {
    group_data <- .x
    group_subjs <- unique(group_data$subj)
    b <- boot(group_subjs, function(d, i) {
      samp <- map_dfr(d[i], ~ group_data %>% filter(subj == .x))
      sd(samp$error)
    }, R = R_boot)
    ci <- boot.ci(b, type = "perc")$percent[4:5]
    tibble(sd = b$t0, ci_low = ci[1], ci_high = ci[2])
  })
condition_sd_boot
#   condition    sd ci_low ci_high
#   <fct>     <dbl>  <dbl>   <dbl>
# 1 noFB       25.8   21.5    29.7
# 2 FB         34.0   29.8    38.0
# => error's location and spread (very close) both depend on condition

# 8. Does error's median change with magnitude of fb_offset?
# Controlled for subjects
fb_data <- eda_data %>%
  filter(condition == "FB") %>%
  mutate(abs_offset = abs(fb_offset))
offset_slope_boot <- boot(subjs, function(d, i) {
  samp <- map_dfr(d[i], ~ fb_data %>% filter(subj == .x))
  coef(rq(error ~ abs_offset, tau = 0.5, data = samp))["abs_offset"]
}, R = R_boot)
boot.ci(offset_slope_boot, type = "perc")
# [0.035 0.081]
# error grows with greater offset => unreliable trials lead to greater error
# participants do not ignore the feedback even when unreliable
# 9. Does error's variance grow with magnitude of fb_offset?
# Controlled for subjects
fb_data <- fb_data %>%
  mutate(abs_offset_bin = cut(abs_offset, breaks = seq(0, 180, by = 20), include.lowest = TRUE))
offset_sd_boot_by_bin <- fb_data %>%
  group_by(abs_offset_bin) %>%
  group_modify(~ {
    bin_data <- .x
    bin_subjs <- unique(bin_data$subj)
    b <- boot(bin_subjs, function(d, i) {
      samp <- map_dfr(d[i], ~ bin_data %>% filter(subj == .x))
      sd(samp$error)
    }, R = R_boot)
    ci <- boot.ci(b, type = "perc")$percent[4:5]
    tibble(sd = b$t0, ci_low = ci[1], ci_high = ci[2])
  })
ggplot(offset_sd_boot_by_bin, aes(x = abs_offset_bin, y = sd)) +
  geom_pointrange(aes(ymin = ci_low, ymax = ci_high)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "|fb_offset| (binned)", y = "SD(error) with 95% bootstrap CI")
# Increases at first and then plateaus at around 60 degrees of offset
# => There is a cut off point to the effect of fb_offset on error's variance

# Thus error's variance needs to be modelled across:
# 1. target (stimuli)
# 2. condition (unisensory or multisensory)
# 3. For multisensory, fb_offset
# Error's location is dependent on both condition and fb_offset as well

# 10. Does |error| (1/accuracy) change with viewAmount (attention to feedback)?
# Plan is to use this property as a validation check for the models
# Controlled for subjects
accuracy_view_amount_slope_boot <- boot(subjs, function(d, i) {
  samp <- map_dfr(d[i], ~ fb_data %>% filter(subj == .x))
  coef(rq(abs(error) ~ viewAmount, tau = 0.5, data = samp))["viewAmount"]
}, R = R_boot)
boot.ci(accuracy_view_amount_slope_boot, type = "perc")
# [0.21 0.77]
# Accuracy reduces as viewAmount grows ie. lower attention paid to the cue

# 11. Does conf change with target (task difficulty)?
# Controlled for subjects
conf_target_slope_boot <- boot(subjs, function(d, i) {
  samp <- map_dfr(d[i], ~ eda_data %>% filter(subj == .x))
  coef(rq(conf ~ target, tau = 0.5, data = samp))["target"]
}, R = R_boot)
boot.ci(conf_target_slope_boot, type = "perc")
# [0.02 0.04]
# 12. Does conf's variance also grow with target?
# Controlled for subjects
conf_sd_boot_by_bin <- eda_data %>%
  group_by(target_bin) %>%
  group_modify(~ {
    bin_data <- .x
    bin_subjs <- unique(bin_data$subj)
    b <- boot(bin_subjs, function(d, i) {
      samp <- map_dfr(d[i], ~ bin_data %>% filter(subj == .x))
      sd(samp$conf)
    }, R = R_boot)
    ci <- boot.ci(b, type = "perc")$percent[4:5]
    tibble(sd = b$t0, ci_low = ci[1], ci_high = ci[2])
  })
ggplot(conf_sd_boot_by_bin, aes(x = target_bin, y = sd)) +
  geom_pointrange(aes(ymin = ci_low, ymax = ci_high)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = "target (binned)", y = "SD(conf) with 95% bootstrap CI")
# All CIs overlap => conf's variance is not dependent on task difficulty
# 13. Does conf track |error| ie. 1/accuracy?
conf_error_slope_boot <- boot(subjs, function(d, i) {
  samp <- map_dfr(d[i], ~ eda_data %>% filter(subj == .x))
  coef(rq(conf ~ abs(error), tau = 0.5, data = samp))["abs(error)"]
}, R = R_boot)
boot.ci(conf_error_slope_boot, type = "perc")
# [-0.001 0.121] => conf does not track accuracy
# Plan is to conf also in model validation

# Model validations properties:
# 1. Accuracy should reduces with viewAmount
# 2. Some equivalent measure to conf for the model increased with target
#    but does not change with model's accuracy

# Next open question a model which tracks internal uncertainity
# 4 models were conceptual tested for fit
# Model 1: Classical Bayesian Causal Inference model, referenced as a possible mechanism of binding in Noppeney
# Model 2: Hierarchical Gaussian Filter, A layered model where each layer tracks how much the earlier layer is varying
# Model 3: Generalized HGF: Extension of HGF, specifically applied to multisensory cue combination simulation
# Model 4: HBPTM: Hierarchical version of classical PTM

# Models 1 and 3 were selected since their definition resembles the variable structure of the task
# Plan is to build both models for a naive observer considering only target
# Then condition added and then fb_offset added

# Model type 1: BCI model
library(gamlss)

# 1. Naive BCI observer: Does not take feedback into consideration
# respond ~ target + sigma, where sigma ~ a + b.target
naive_bci_formula <- respond ~ offset(target) # Uninformative prior, with respond's mean == target
naive_bci_sigma_formula <- ~ target # SD ie. noise modelled against target
naive_bci_nu_formula <- ~ 1 # Right skewed distribution for overall model
naive_bci_family <- SN1() # Right normal distribution with only one parameter for skewness
# Fit the model individually for each subject
naive_bci_fits <- eda_data %>%
  select(-fb_validity) %>%
  group_split(subj) %>%
  set_names(levels(eda_data$subj)) %>%
  map(~ gamlss(
                formula = naive_bci_formula,
                sigma.formula = naive_bci_sigma_formula,
                nu.formula = naive_bci_nu_formula,
                family = naive_bci_family,
                data = .x,
                control = gamlss.control(n.cyc = 300)
  ))
# All subjects converged after increasing the cycles from default to 100 -> 200 -> 300

# Condition aware BCI model: Treats feedback and no feedback information differently
condition_bci_formula <- respond ~ offset(target) + condition # condition added as a predictor
condition_bci_sigma_formula <- ~ target + condition # SD ie. noise modelled against target and condition
condition_bci_nu_formula <- ~ 1
condition_bci_family <- SN1()
# Fit the model individually for each subject
condition_bci_fits <- eda_data %>%
  select(-fb_validity) %>%
  group_split(subj) %>%
  set_names(levels(eda_data$subj)) %>%
  map(~ gamlss(
                formula = condition_bci_formula,
                sigma.formula = condition_bci_sigma_formula,
                nu.formula = condition_bci_nu_formula,
                family = condition_bci_family,
                data = .x,
                control = gamlss.control(n.cyc = 300)
  ))
# All subjects converged at 300 cycles

# Feedback reliability aware BCI model: Weighs whether the feedback is useful or not based on it's |offset|
feedback_bci_formula <- respond ~ offset(target) + condition + condition:abs(fb_offset)
# offset magnitude added as a predictor
feedback_bci_sigma_formula <- ~ target + condition + condition:abs(fb_offset)
# SD ie. noise modelled against target, condition and |offset|
feedback_bci_nu_formula <- ~ 1
feedback_bci_family <- SN1()
# Fit the model individually for each subject
feedback_bci_fits <- eda_data %>%
  select(-fb_validity) %>%
  group_split(subj) %>%
  set_names(levels(eda_data$subj)) %>%
  map(~ gamlss(
                formula = feedback_bci_formula,
                sigma.formula = feedback_bci_sigma_formula,
                nu.formula = feedback_bci_nu_formula,
                family = feedback_bci_family,
                data = .x,
                control = gamlss.control(n.cyc = 300)
  ))
# s10's nu (distribution shape) does not converge
# Need to think of a solution for this
# Turns out including all the information, and not filtering
# for FB trials, allowed s10 fit to converge

# Compute the BIC for each model across all subjects
bic_naive <- imap_dfr(naive_bci_fits, ~ tibble(subj = .y, bic = AIC(.x, k = log(nobs(.x)))))
bic_condition <- imap_dfr(condition_bci_fits, ~ tibble(subj = .y, bic = AIC(.x, k = log(nobs(.x)))))
bic_feedback <- imap_dfr(feedback_bci_fits, ~ tibble(subj = .y, bic = AIC(.x, k = log(nobs(.x)))))

# Join the fits into one large tibble
bic_by_subj <- bic_naive %>%
  rename(bic_naive = bic) %>%
  left_join(bic_condition %>% rename(bic_condition = bic), by = "subj") %>%
  left_join(bic_feedback %>% rename(bic_feedback = bic), by = "subj") %>%
  mutate(
  delta_naive = bic_naive - bic_feedback, # relative to feedback_bci, most complex model
  delta_condition = bic_condition - bic_feedback # relative to feedback_bci, most complex model
)

# Compute mean + se by summarizing across subjects
bic_delta_summary <- bic_by_subj %>%
  select(subj, delta_naive, delta_condition) %>%
  pivot_longer(-subj, names_to = "model", values_to = "delta_bic") %>%
  mutate(model = factor(model,
                         levels = c("delta_naive", "delta_condition"),
                         labels = c("naive", "condition"))) %>%
  group_by(model) %>%
  summarise(mean_delta = mean(delta_bic),
            se_delta = sd(delta_bic) / sqrt(n()),
            .groups = "drop")

# Plot to compare how much each model was able to explain the stimulus information
bic_delta_summary %>%
ggplot(aes(x = model, y = mean_delta)) +
  geom_pointrange(aes(ymin = mean_delta - se_delta, ymax = mean_delta + se_delta)) +
  geom_hline(yintercept = 0) +
  labs(x = "model", y = "mean delta BIC vs fb offset magnitude added model (+/- SE)") +
  theme_bw()
# Both are positive => feedback_bci is the best model
# But both condition and feedback model are difficult to connect behavioral uncertainity
# Both condition and feedback models need to be modified so as to create competing uncentainities