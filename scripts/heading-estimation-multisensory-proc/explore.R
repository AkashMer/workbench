# Initialize required libraries
library(tidyverse)
library(here)
library(circular)

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
# Key takeway: The noise in perception may need to be modeled by both condition and fb_validity

# Below this line, the code needs to be streamline with the new approach of going down the
# column headings

# Error distribuion across both conditions
error_summary <- bva_dataconf %>%
  group_by(condition) %>%
  summarise(mean_val = mean(error), median_val = median(error))

ggplot(bva_dataconf, aes(x = error)) +
  geom_histogram(aes(y = after_stat(density))) +
  geom_density() +
  geom_vline(data = error_summary, aes(xintercept = mean_val), color = "red", linetype = "dashed") +
  geom_vline(data = error_summary, aes(xintercept = median_val), color = "green", linetype = "dashed") +
  facet_wrap(~condition)
# Both errors are right skewed with slightly shifted centers away from zero error
# But the FB trials are more right skewed
# This suggests that participants generally overestimated the target
# Could this be due to the validity split of the fb_offset
# Since validity labels are not present, I will use label valid for gaussian (0, 30) +/- 3SD
plot_data <- bva_dataconf %>%
  mutate(cue_validity = if_else(abs(fb_offset) <= 90, "valid", "nonvalid"))

validity_summary <- plot_data %>%
  group_by(condition, cue_validity) %>%
  summarise(mean_val = mean(error), median_val = median(error), .groups = "drop")

ggplot(plot_data, aes(x = error)) +
  geom_histogram(aes(y = after_stat(density))) +
  geom_density() +
  geom_vline(data = validity_summary, aes(xintercept = mean_val), color = "red", linetype = "dashed") +
  geom_vline(data = validity_summary, aes(xintercept = median_val), color = "green", linetype = "dashed") +
  facet_grid(cue_validity ~ condition)
# Still slghtly right skewed but much less than FB trials with valid cues

# How does error relate to target value?
# Bin the target into 10 bins of 36 degrees each
plot_data <- plot_data %>%
  mutate(target_bin = cut(target, breaks = 10))
# Plot across both condition and cue validity
ggplot(plot_data, aes(x = target_bin, y = error)) +
  geom_boxplot() +
  facet_grid(cue_validity ~ condition) +
  theme(axis.text.x = element_blank()) +
  labs(x = "target (binned, low -> high)", y = "error")
# Error's variance grows with target angle
# => the variance of error is a function of target angle
# ie. larger the self-motion movement, larger errors
# => less reliability of the proprioceptory/vestibular senses for larger movements
# Thus, a naive forced fusion model is not the right choice here

# Subject level distribution of error variable
subj_plot_data <- plot_data %>%
  filter(condition == "noFB" | cue_validity == "valid")

ggplot(subj_plot_data, aes(x = error, fill = condition)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~subj, ncol = 6)
# Check the corresponding means and medians along the subjects
subj_mean <- subj_plot_data %>%
  group_by(subj, condition) %>%
  summarise(val = mean(error), .groups = "drop")
subj_median <- subj_plot_data %>%
  group_by(subj, condition) %>%
  summarise(val = median(error), .groups = "drop")

ggplot(subj_mean, aes(x = subj, y = val, color = condition, group = condition)) +
  geom_point() + geom_line()
ggplot(subj_median, aes(x = subj, y = val, color = condition, group = condition)) +
  geom_point() + geom_line()
# Confirms the right skewness and also that FB trials have greater errors across all subjects
# even when the cue validity is good

# Check if a similar pattern holds for noFB vs uniformly derived validity
subj_plot_nonvalid_data <- plot_data %>%
  filter(condition == "noFB" | cue_validity == "nonvalid")

subj_nonvalid_mean <- subj_plot_nonvalid_data %>%
  group_by(subj, condition) %>%
  summarise(val = mean(error), .groups = "drop")
subj_nonvalid_median <- subj_plot_nonvalid_data %>%
  group_by(subj, condition) %>%
  summarise(val = median(error), .groups = "drop")

ggplot(subj_nonvalid_mean, aes(x = subj, y = val, color = condition, group = condition)) +
  geom_point() + geom_line()
ggplot(subj_nonvalid_median, aes(x = subj, y = val, color = condition, group = condition)) +
  geom_point() + geom_line()
# Similar pattern holds except fro s9; most likely from low trial count
count_data <- plot_data %>%
  mutate(group = if_else(condition == "noFB", "noFB", cue_validity)) %>%
  count(subj, group)

ggplot(count_data, aes(x = subj, y = n, color = group, group = group)) +
  geom_point() + geom_line()
# s5, s9 and s21 have low trials across all types of trials

# Confidence spread analysis
# Confirm all trials had non-zero conf values
plot_data %>% filter(conf == 0) # 6, most in s14 for low error values
# Get the same relationship of mean of conf across subjects and trial types
conf_mean <- subj_plot_data %>%
  group_by(subj, condition) %>%
  summarise(val = mean(conf), .groups = "drop")
conf_median <- subj_plot_data %>%
  group_by(subj, condition) %>%
  summarise(val = median(conf), .groups = "drop")

ggplot(conf_mean, aes(x = subj, y = val, color = condition, group = condition)) +
  geom_point() + geom_line()
ggplot(conf_median, aes(x = subj, y = val, color = condition, group = condition)) +
  geom_point() + geom_line()
# and median
conf_nonvalid_mean <- subj_plot_nonvalid_data %>%
  group_by(subj, condition) %>%
  summarise(val = mean(conf), .groups = "drop")
conf_nonvalid_median <- subj_plot_nonvalid_data %>%
  group_by(subj, condition) %>%
  summarise(val = median(conf), .groups = "drop")

ggplot(conf_nonvalid_mean, aes(x = subj, y = val, color = condition, group = condition)) +
  geom_point() + geom_line()
ggplot(conf_nonvalid_median, aes(x = subj, y = val, color = condition, group = condition)) +
  geom_point() + geom_line()
# Not a lot of difference, both track similarly across all subjects
# Slightly lower confidence (higher conf) in case of nonvalid cues
# Need to compare valid vs nonvalid cues
validity_conf_mean <- plot_data %>%
  group_by(subj, cue_validity) %>%
  summarise(val = mean(conf), .groups = "drop")
validity_conf_median <- plot_data %>%
  group_by(subj, cue_validity) %>%
  summarise(val = median(conf), .groups = "drop")

ggplot(validity_conf_mean, aes(x = subj, y = val, color = cue_validity, group = cue_validity)) +
  geom_point() + geom_line()
ggplot(validity_conf_median, aes(x = subj, y = val, color = cue_validity, group = cue_validity)) +
  geom_point() + geom_line()
# Mean conf are higher ie. lower confidence for non-valid vs valid
# But medians do not show a clear separation, possible outliers?
ggplot(plot_data, aes(x = conf, fill = cue_validity)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~cue_validity)
# The distribution of nonvalid trials for conf is much more spread out
# Suggesting an effect of outliers on means, which the medians are not sensitive to
# Also, conf shows an interesting bump in the right tail for valid cue trials
# Confirm against noFB trials
ggplot(plot_data, aes(x = conf, fill = condition)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~condition)
# Even those shows a smaller bump in the right tail
# Needs more exploration regarding this phenomenon at subject level
ggplot(subj_plot_data, aes(x = conf, fill = condition)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~subj, ncol = 6)
# Several subjects show higher frequency in the 30-40 range while majority show higher frequency in 0-10
# s2, s7, s9, s10, s12, s13, s21, s24, s31

# Trial wise analysis of error, conf and trial_duration
# Check the trial variable distribution
plot_data %>%
  count(subj, trial) %>%
  ggplot(aes(x = trial, y = subj, fill = n)) +
  geom_tile()
# s4, s5, s7, s10: Truncated session
# s9, s14, s21 (highest), s24: Scattered dropped trials
# Some trials might have been excluded from analysis
# Trial wise overall error, conf and trial_duration
ggplot(plot_data, aes(x = trial, y = error)) + geom_smooth()
ggplot(plot_data, aes(x = trial, y = conf)) + geom_smooth()
ggplot(plot_data, aes(x = trial, y = trial_duration)) + geom_smooth()
# No trend across trials
ggplot(plot_data, aes(x = trial, y = error, color = condition)) + geom_smooth()
ggplot(plot_data, aes(x = trial, y = conf, color = condition)) + geom_smooth()
ggplot(plot_data, aes(x = trial, y = trial_duration, color = condition)) + geom_smooth()
# No trend but sessions were divided into noFB trials flanking the FB trials in the middle
fb_only <- plot_data %>% filter(condition == "FB")
ggplot(fb_only, aes(x = trial, y = error, color = cue_validity)) + geom_smooth()
ggplot(fb_only, aes(x = trial, y = conf, color = cue_validity)) + geom_smooth()
ggplot(fb_only, aes(x = trial, y = trial_duration, color = cue_validity)) + geom_smooth()
# Again no trend, but fb_offset assignment was random across the FB trials
# Confirm this pattern holds at subject level as well
ggplot(bva_dataconf, aes(x = trial, y = error)) +
  geom_line() + facet_wrap(~subj, ncol = 6) # No trend even at subject level
ggplot(bva_dataconf, aes(x = trial, y = conf)) +
  geom_line() + facet_wrap(~subj, ncol = 6) # No trial order trend, but differing min/max
# most likely points toward temperament of individual subjects
ggplot(bva_dataconf, aes(x = trial, y = trial_duration)) +
  geom_line() + facet_wrap(~subj, ncol = 6) # No trend even at subject level

# Trial duration analysis by condition
# Compare trial duration across the three conditions
plot_data %>%
  mutate(group = if_else(condition == "noFB", "noFB", cue_validity)) %>%
  ggplot(aes(x = trial_duration)) +
  geom_histogram(aes(y = after_stat(density))) +
  geom_density() +
  facet_wrap(~group)
# All three are similar with peaks around 4000-5000 msec and right skewed
# Compute post feedback response time to check how it distributes across valid and nonvalid trials
plot_data <- plot_data %>%
  mutate(post_fb_rt = trial_duration - fb_time)
# Compare distribution across both cue validities
plot_data %>%
  filter(condition == "FB") %>%
  ggplot(aes(x = post_fb_rt, fill = cue_validity)) +
  geom_density(alpha = 0.4) +
  facet_wrap(~cue_validity)
# Post-feedback response time does not depend on cue validity

# viewAmount Analysis
fb_data <- plot_data %>%
  mutate(viewAmount_bin = cut(viewAmount, breaks = 10))

viewAmount_duration <- fb_data %>%
  group_by(viewAmount_bin) %>%
  summarise(mean_val = mean(trial_duration), se = sd(trial_duration) / sqrt(n()))
viewAmount_target <- fb_data %>%
  group_by(viewAmount_bin) %>%
  summarise(mean_val = mean(target), se = sd(target) / sqrt(n()))
viewAmount_error <- fb_data %>%
  group_by(viewAmount_bin) %>%
  summarise(mean_val = mean(error), se = sd(error) / sqrt(n()))
viewAmount_by_validity <- fb_data %>%
  group_by(cue_validity) %>%
  summarise(mean_val = mean(viewAmount), se = sd(viewAmount) / sqrt(n()))

ggplot(viewAmount_duration, aes(x = viewAmount_bin, y = mean_val)) +
  geom_pointrange(aes(ymin = mean_val - se, ymax = mean_val + se)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(viewAmount_target, aes(x = viewAmount_bin, y = mean_val)) +
  geom_pointrange(aes(ymin = mean_val - se, ymax = mean_val + se)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(viewAmount_error, aes(x = viewAmount_bin, y = mean_val)) +
  geom_pointrange(aes(ymin = mean_val - se, ymax = mean_val + se)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplot(viewAmount_by_validity, aes(x = cue_validity, y = mean_val)) +
  geom_pointrange(aes(ymin = mean_val - se, ymax = mean_val + se))
# trial_duration is inversely proportional which makes sense based on how viewAmount is defined
# Error and cue_validity do not so much difference/trend
# Lower targets have lower viewAmount is interesting
ggplot(fb_data, aes(x = target, y = viewAmount)) +
  geom_point(alpha = 0.3)
# This shows lower targets, the viewAmount is constrained by the target (grouping)
# But for higher targets, viewAmount plateus instead of showing any trend

# To summarize:
# error vs condition : FB > noFB, both right-skewed
# error vs cue_validity : nonvalid closer to noFB
# error vs target (binned) : variance increases with target
# error vs subj by condition : FB error consistently above noFB per subject (valid cues); inconsistent for nonvalid
# subject-wise conf : distinct per-subject => temperament

# Confirm the relationship between SD(error) and target
# ie. the relationship is plain linear or not
target_sd <- plot_data %>%
  group_by(target_bin) %>%
  summarise(sd_val = sd(error))

ggplot(target_sd, aes(x = target_bin, y = sd_val)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Fairly linear
# Across conditions?
condition_sd <- plot_data %>%
  group_by(condition, cue_validity, target_bin) %>%
  summarise(sd_val = sd(error), .groups = "drop")

ggplot(condition_sd, aes(x = target_bin, y = sd_val)) +
  geom_point() +
  facet_grid(cue_validity ~ condition) +
  theme(axis.text.x = element_blank()) +
  labs(x = "target (binned, low -> high)", y = "SD(error)")
# Similarly linear, but less straight for low validity data
# Confirm the exact relationship by fitting a line
# Compute the SD(error) across target bins
sd_summary <- plot_data %>%
  group_by(target_bin) %>%
  summarise(target_mid = mean(target), sd_error = sd(error))

fit <- lm(sd_error ~ target_mid, data = sd_summary)
slope_lin <- coef(fit)[["target_mid"]]
intercept_lin <- coef(fit)[["(Intercept)"]]

ggplot(sd_summary, aes(x = target_mid, y = sd_error)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(x = "target", y = "SD(error)",
       title = paste0("slope = ", round(slope_lin, 3), ", intercept = ", round(intercept_lin, 2)))
# SD(error) intercept = 16.33, which is close to minimum value for SD(error)
# => SD(error) has a minimal value but beyond that it linearly varies with target
