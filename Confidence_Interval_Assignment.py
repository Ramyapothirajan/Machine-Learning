import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

cars = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/15. Confidence_Interval/Cars.csv")

# Parameters for the normal distribution
mean = cars.MPG.mean()
std_dev = cars.MPG.std()

# Calculate the probability of MPG of Cars for the below cases.

# a.P(MPG>38)
# Calculate the probability of MPG > 38 using the cumulative distribution function (CDF) of the normal distribution
probability_greater_than_38 = 1 - stats.norm.cdf(38,mean, std_dev)
print("Probability of MPG > 38:", probability_greater_than_38)

# b.P(MPG<40)
# Calculate the probability of MPG < 40 using the cumulative distribution function (CDF) of the normal distribution
probability_less_than_40 = stats.norm.cdf(40, mean, std_dev)
print("Probability of MPG < 40:", probability_less_than_40)

# c.P(20<MPG<50)
# Probability of MPG less than 50
prob_less_than_50 = stats.norm.cdf(50, mean, std_dev)

# Probability of MPG less than 20
prob_less_than_20 = stats.norm.cdf(20, mean, std_dev)

# Probability of MPG between 20 and 50
probability_between_20_and_50 = prob_less_than_50 - prob_less_than_20
print("Probability of 20 < MPG < 50:", probability_between_20_and_50)

# Q2) Check whether the data follows the normal distribution.
# a) Check whether the MPG of Cars follows the Normal Distribution Dataset: Cars.csv

# Extract MPG data
mpg_data = cars['MPG']

# Plot a histogram to visualize the distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(mpg_data, bins=20, edgecolor='black')
plt.title('Histogram of MPG')
plt.xlabel('MPG')
plt.ylabel('Frequency')

# Q-Q plot for normality assessment
plt.subplot(1, 2, 2)
stats.probplot(mpg_data, dist="norm", plot=plt)
plt.title('Q-Q plot of MPG against Normal Distribution')

plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
shapiro_test_stat, shapiro_p_value = stats.shapiro(mpg_data)
alpha = 0.05
if shapiro_p_value > alpha:
    print(f'Shapiro-Wilk Test: Sample looks normally distributed (fail to reject H0), p-value={shapiro_p_value:.4f}')
else:
    print(f'Shapiro-Wilk Test: Sample does not look normally distributed (reject H0), p-value={shapiro_p_value:.4f}')

# b) Check Whether the Adipose Tissue (AT) and Waist Circumference (Waist) from wc-at data set follow Normal Distribution
wc_at = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/15. Confidence_Interval/wc-at.csv")

# Extract 'Adipose Tissue (AT)' and 'Waist Circumference (Waist)' data
at_data = wc_at['AT']
waist_data = wc_at['Waist']

# Plot histograms for visual assessment
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(at_data, bins=20, edgecolor='black')
plt.title('Histogram of Adipose Tissue (AT)')
plt.xlabel('AT')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(waist_data, bins=20, color='green', edgecolor='black')
plt.title('Histogram of Waist Circumference (Waist)')
plt.xlabel('Waist')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Q-Q plot for normality assessment
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
stats.probplot(at_data, dist="norm", plot=plt)
plt.title('Q-Q plot of Adipose Tissue (AT) against Normal Distribution')

plt.subplot(1, 2, 2)
stats.probplot(waist_data, dist="norm", plot=plt)
plt.title('Q-Q plot of Waist Circumference (Waist) against Normal Distribution')

plt.tight_layout()
plt.show()

# Shapiro-Wilk test for normality
def shapiro_test_and_print(data, name):
    shapiro_test_stat, shapiro_p_value = stats.shapiro(data)
    alpha = 0.05
    if shapiro_p_value > alpha:
        print(f'{name}: Sample looks normally distributed (fail to reject H0), p-value={shapiro_p_value:.4f}')
    else:
        print(f'{name}: Sample does not look normally distributed (reject H0), p-value={shapiro_p_value:.4f}')

shapiro_test_and_print(at_data, 'Adipose Tissue (AT)')
shapiro_test_and_print(waist_data, 'Waist Circumference (Waist)')

# Q3) Calculate the Z scores of 90% confidence interval,94% confidence interval, and 60% confidence interval.
z_value_90 = stats.norm.ppf(0.90, 0, 1)  # Probability of 0.95 for a one-tailed 90% confidence interval
print(f"Z-value for 95% confidence interval: {z_value_90:.4f}")

z_value_94 = stats.norm.ppf(0.94, 0, 1)  # Probability of 0.95 for a one-tailed 94% confidence interval
print(f"Z-value for 95% confidence interval: {z_value_94:.4f}")

z_value_60 = stats.norm.ppf(0.60, 0, 1)  # Probability of 0.95 for a one-tailed 60% confidence interval
print(f"Z-value for 95% confidence interval: {z_value_60:.4f}")

# Q4) Calculate the t scores of 95% confidence interval, 96% confidence interval, and 99% confidence interval for the sample size of 25.
df = 25 - 1  # degrees of freedom
t_score_95 = stats.t.ppf(0.975, df)  # t-score for 95% confidence interval with 24 degrees of freedom
print(f"t-score for a 95% confidence interval with 24 degrees of freedom: {t_score_95:.4f}")

t_score_96 = stats.t.ppf(0.98, df)  # t-score for 96% confidence interval with 24 degrees of freedom
print(f"t-score for a 96% confidence interval with 24 degrees of freedom: {t_score_96:.4f}")

t_score_99 = stats.t.ppf(0.995, df)  # t-score for 99% confidence interval with 24 degrees of freedom
print(f"t-score for a 99% confidence interval with 24 degrees of freedom: {t_score_99:.4f}")

''' Q5) A Government company claims that an average light bulb lasts 270 days. A researcher randomly selects 18 bulbs for testing. 
The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. 
If the CEO's claim were true, what is the probability that 18 randomly selected bulbs would have an average life of no more than 260 days?
'''
import scipy.stats as stats

mu = 270  # Population mean claim
x_bar = 260  # Sample mean
sd = 90  # Sample standard deviation
n = 18  # Sample size

t_score = (x_bar - mu) / (sd / (n ** 0.5))
print("t-score:", t_score)

# Calculate the probability using the t-distribution CDF
probability = stats.t.cdf(t_score, n - 1)
print("Probability:", probability)

# This probability value represents the likelihood that a sample of 18 bulbs would have an average life of no more than 260 days if the CEO's claim (population mean of 270 days) were true.

''' Q6) The time required for servicing transmissions is normally distributed between  = 45 minutes and  = 8 minutes. 
The service manager plans to have work begin on the transmission of a customer’s car 10 minutes after the car is dropped off and the customer is told that the car will be ready within 1 hour from drop-off. 
What is the probability that the service manager cannot meet his commitment?
'''
import scipy.stats as stats

mu = 45  # Mean service time
sigma = 8  # Standard deviation

# Calculate the Z-score for 60 minutes (1 hour)
x = 50  # Remaining time for servicing after work begins (60 - 10)
Z = (x - mu) / sigma

# Find the probability using the normal distribution CDF
probability = 1 - stats.norm.cdf(Z)
print("Probability of not meeting the commitment:", probability)


''' Q7) The current age (in years) of 400 clerical employees at an insurance claims processing center is normally distributed with mean  = 38 and 
Standard deviation=6. For each statement below, please specify True/False. If false, briefly explain why.
'''

# A. More employees at the processing center are older than 44 than between 38 and 44.
import scipy.stats as stats

# Given parameters
mu = 38  # Mean
sigma = 6  # Standard deviation

# Calculate Z-scores for 44 and 38 years
Z_44 = (44 - mu) / sigma
Z_38 = (38 - mu) / sigma

# Probability an employee is older than 44
prob_older_than_44 = 1 - stats.norm.cdf(Z_44)

# Probability an employee's age is between 38 and 44
prob_between_38_44 = stats.norm.cdf(Z_44) - stats.norm.cdf(Z_38)

print("Probability an employee is older than 44 years:", prob_older_than_44)
print("Probability an employee's age is between 38 and 44 years:", prob_between_38_44)

# Comparison for Statement A
if prob_older_than_44 > prob_between_38_44:
    print("Statement A: True - More employees are older than 44 than between 38 and 44.")
else:
    print("Statement A: False - More employees are not older than 44 than between 38 and 44.")

# B. A training program for employees under the age of 30 at the center would be expected to attract about 36 employees.
# Calculate Z-score for 30 years
Z_30 = (30 - mu) / sigma

# Probability an employee's age is under 30
prob_under_30 = stats.norm.cdf(Z_30)

# Expected count for employees under the age of 30
expected_count_under_30 = prob_under_30 * 400

print("Probability an employee is under the age of 30:", prob_under_30)
print("Expected count for employees under the age of 30:", expected_count_under_30)

# Comparison for Statement B
if 35 < expected_count_under_30 < 37:
    print("Statement B: True - The expected count is around 36.")
else:
    print("Statement B: False - The expected count is not around 36.")

''' Q9) Let X ~ N(100, 20^2) its (100, 20 square). Find two values, a and b, symmetric about the mean, 
such that the probability of the random variable taking a value between them is 0.99.
'''
import scipy.stats as stats

# Given parameters
mean = 100
std_dev = 20

# Calculate the z-value for 99% of the data lying between a and b
z_value = stats.norm.ppf(0.995)  # 99.5% area to capture 99% between two tails

# Calculate a and b
a = mean - z_value * std_dev
b = mean + z_value * std_dev

print(f"The value of 'a' is approximately: {a:.2f}")
print(f"The value of 'b' is approximately: {b:.2f}")

''' Q10) Consider a company that has two different divisions. 
The annual profits from the two divisions are independent and have distributions Profit1 ~ N(5, 3^2) and Profit2 ~ N(7, 4^2) respectively. 
Both the profits are in $ Million. Answer the following questions about the total profit of the company in Rupees. 
Assume that $1 = Rs. 45
'''
# Given means and variances
mean_profit1 = 5  # in $ Million
mean_profit2 = 7  # in $ Million
variance_profit1 = 3 ** 2
variance_profit2 = 4 ** 2

# Calculate Mean Total Profit
mean_total_profit = mean_profit1 + mean_profit2

# Calculate Variance Total Profit
variance_total_profit = variance_profit1 + variance_profit2

# Calculate Standard Deviation of Total Profit
std_dev_total_profit = (variance_total_profit) ** 0.5

# Conversion Rate
conversion_rate = 45  # $1 = Rs. 45

# Convert Total Profit from $ to Rs
total_profit_rs = mean_total_profit * conversion_rate

print(f"The total profit of the company in Rupees is: {total_profit_rs} Rs.")

# A. Specify a Rupee range (centered on the mean) such that it contains 95% probability for the annual profit of the company.

# Z-score for 95% confidence interval
z_score_95 = 1.96

# Calculate Margin of Error
margin_of_error = z_score_95 * std_dev_total_profit

# Calculate Lower and Upper Range for 95% Confidence Interval
lower_range = mean_total_profit - margin_of_error
upper_range = mean_total_profit + margin_of_error

# Convert the ranges to Rupees
lower_range_rs = lower_range * conversion_rate
upper_range_rs = upper_range * conversion_rate

print(f"The 95% confidence interval for the total annual profit in Rupees is approximately: {lower_range_rs:.2f} Rs to {upper_range_rs:.2f} Rs")

# This code calculates the 95% confidence interval for the total annual profit of the company in Rupees. 
# It uses the Z-score of 1.96 for a 95% confidence interval and the provided mean profits and variances.

# B. Specify the 5th percentile of profit (in Rupees) for the company.

# Calculate the 5th percentile value for the total profit in $ (Million)
fifth_percentile_profit = stats.norm.ppf(0.05, mean_total_profit, std_dev_total_profit)

# Convert the 5th percentile value to Rupees
fifth_percentile_profit_rs = fifth_percentile_profit * conversion_rate

print(f"The 5th percentile of profit for the company in Rupees is approximately: {fifth_percentile_profit_rs:.2f} Rs")


# C. Which of the two divisions has a larger probability of making a loss each year?
std_dev_profit1 = 3  # standard deviation for division 1
std_dev_profit2 = 4  # standard deviation for division 2

# Calculate Z-score for zero profit (loss)
z_score_loss_div1 = (0 - mean_profit1) / std_dev_profit1
z_score_loss_div2 = (0 - mean_profit2) / std_dev_profit2

# Calculate the probability of a loss using the standard normal distribution
probability_loss_div1 = stats.norm.cdf(z_score_loss_div1)
probability_loss_div2 = stats.norm.cdf(z_score_loss_div2)

print(f"Probability of a loss for Division 1: {probability_loss_div1:.4f}")
print(f"Probability of a loss for Division 2: {probability_loss_div2:.4f}")


''' 3.	Suppose we want to estimate the average weight of an adult male in Mexico. 
We draw a random sample of 2,000 men from a population of 3,000,000 men and weigh them. 
We find that the average person in our sample weighs 200 pounds, and the standard deviation of the sample is 30 pounds. 
Calculate 94%,98%,96% confidence interval?
'''
import scipy.stats as stats

# Given data
sample_mean = 200  # Sample mean
sample_std_dev = 30  # Sample standard deviation
sample_size = 2000  # Sample size

# Calculate standard error (standard deviation of the sampling distribution of the sample mean)
standard_error = sample_std_dev / (sample_size ** 0.5)

# Calculate the Z-scores for different confidence levels
z_scores = {
    94: stats.norm.ppf(0.5 + 0.94 / 2),
    98: stats.norm.ppf(0.5 + 0.98 / 2),
    96: stats.norm.ppf(0.5 + 0.96 / 2)
}

# Calculate confidence intervals
confidence_intervals = {}
for confidence_level, z_score in z_scores.items():
    margin_of_error = z_score * standard_error
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    confidence_intervals[confidence_level] = (lower_bound, upper_bound)

# Display confidence intervals
for confidence_level, interval in confidence_intervals.items():
    print(f"{confidence_level}% Confidence Interval: {interval}")
