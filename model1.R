library(lavaan)
library(dplyr)

# Read the data
setwd('/home/veeraruuskanen/research/[22-23-1.2] eeg-pupil-detection')
data <- read.csv("data-pupil-detection-eeg.csv")
data$response_numeric <- ifelse(data$response == 'space', 1, 0)

# Split the data by subject_nr 
subject_data <- split(data, data$subject_nr)

# Split by time (i.e., first and second half of exp; for control analysis)
first_half_all <- data.frame()
second_half_all <- data.frame()

for (subject_nr in names(subject_data)) {
  subject_x <- subject_data[[subject_nr]]
  total_trials <- nrow(subject_x)
  
  first_half <- subject_x[1:(total_trials / 2), ]
  second_half <- subject_x[((total_trials / 2) + 1):total_trials, ]
  
  first_half_all <- rbind(first_half_all, first_half)
  second_half_all <- rbind(second_half_all, second_half)
}

subject_data_first <- split(first_half_all, first_half_all$subject_nr)
subject_data_second <- split(second_half_all, second_half_all$subject_nr)

# Function to run lavaan model for each subject
run_model <- function(subject_data) {
  model <- '
    response ~ b11*mean_alpha + b12*mean_beta + b13*mean_theta + c11*mean_pupil

    # mediator regression
    mean_alpha ~ a11*mean_pupil
    mean_beta ~ a21*mean_pupil
    mean_theta ~ a31*mean_pupil

    # mediator residual covariance
    mean_alpha ~~ mean_beta
    mean_alpha ~~ mean_theta
    mean_beta ~~ mean_theta

    # effect decomposition
    # y1 ~ x1
    alpha_response := b11
    beta_response := b12
    theta_response := b13
    pupil_alpha := a11
    pupil_beta := a21
    pupil_theta := a31
    
    # indirect effects
    pupil_alpha_response := a11*b11
    pupil_beta_response := a21*b12
    pupil_theta_response := a31*b13
    pupil_response := c11
  '
  
  fit <- sem(model, data = subject_data)
  path <- sprintf('mediation-output-response/subject%d.csv', subject_data$subject_nr[1])
  print(path)
  estimates <- parameterEstimates(fit)[, -3]
  write.csv(estimates, path)
  return(estimates)
}

# Run the model for each subject and store results
results <- lapply(subject_data, run_model)

# Combine the results
combined_results <- do.call(rbind, results)
write.csv(combined_results, 'mediation-output/combined-results-response.csv')

# Calculate the average estimates and perform one-sample t-test
test_results <- aggregate(combined_results[, c("est", "se")],
                          by = list(combined_results[, "lhs"]),
                          function(x) c(mean = mean(x), t = t.test(x, mu = 0)$statistic, p = t.test(x, mu = 0)$p.value))

# Print the test results
print(test_results)
write.csv(test_results, 'mediation-output/average-response.csv')
