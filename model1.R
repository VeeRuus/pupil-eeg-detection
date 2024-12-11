library(lavaan)
library(dplyr)

# Read the data
setwd('/home/veeraruuskanen/research/ongoing-projects/eeg-pupil-detection/revision')
data <- read.csv("data-clean.csv", sep=",", header = TRUE)
data$response_numeric <- ifelse(data$response == 'space', 1, 0)

# Split the data by subject_nr 
subject_data <- split(data, data$subject_nr)

# Split by time (i.e., first and second half of exp; for potential control analysis)

# first_half_all <- data.frame()
# second_half_all <- data.frame()
#for (subject_nr in names(subject_data)) {
#  subject_x <- subject_data[[subject_nr]]
#  total_trials <- nrow(subject_x)
  
# first_half <- subject_x[1:(total_trials / 2), ]
# second_half <- subject_x[((total_trials / 2) + 1):total_trials, ]
  
# first_half_all <- rbind(first_half_all, first_half)
# second_half_all <- rbind(second_half_all, second_half)
#}

#subject_data_first <- split(first_half_all, first_half_all$subject_nr)
#subject_data_second <- split(second_half_all, second_half_all$subject_nr)

# Function to run lavaan model for each subject (pay attention to dependent variable [correct or response])
run_model <- function(subject_data) {
  model <- '
    response ~ b11*mean_iaf + b12*mean_beta + b13*mean_theta + c11*mean_pupil_z 

    # mediator regression
    mean_iaf ~ a11*mean_pupil_z
    mean_beta ~ a21*mean_pupil_z
    mean_theta ~ a31*mean_pupil_z
    

    # mediator residual covariance
    mean_iaf ~~ mean_beta
    mean_iaf ~~ mean_theta
    mean_beta ~~ mean_theta

    # effect decomposition
    # y1 ~ x1
    iaf_response := b11
    beta_response := b12
    theta_response := b13
    pupil_iaf := a11
    pupil_beta := a21
    pupil_theta := a31
    pupil_response := c11
    
    # indirect effects
    pupil_iaf_response := a11*b11
    pupil_beta_response := a21*b12
    pupil_theta_response := a31*b13
    
  '
  
  fit <- sem(model, data = subject_data, ordered=c("correct")) # ordered = communicates to lavaan that the variable is categorical (binary)
  path <- sprintf('mediation-output-clean-response/subject%d.csv', subject_data$subject_nr[1])
  print(path)
  estimates <- parameterEstimates(fit)[, -3]
  write.csv(estimates, path)
  return(estimates)
}

# Run the model for each subject and store results
results <- lapply(subject_data, run_model)

# Combine the results
combined_results <- do.call(rbind, results)
write.csv(combined_results, 'mediation-output-clean-response/combined-results-response.csv')

# Calculate the average estimates and perform one-sample t-test
test_results <- aggregate(combined_results[, c("est", "se")],
                          by = list(combined_results[, "lhs"]),
                          function(x) c(mean = mean(x), t = t.test(x, mu = 0)$statistic, p = t.test(x, mu = 0)$p.value))

# Print the test results
print(test_results)
write.csv(test_results, 'mediation-output-clean-response/average-response.csv')
