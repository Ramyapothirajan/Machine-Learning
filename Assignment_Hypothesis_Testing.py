import pandas as pd
import numpy as np
import scipy
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


'''1. A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. 
A randomly selected sample of cutlets was collected from both units and measured. 
Analyze the data and draw inferences at a 5% significance level. 
Please state the assumptions and tests that you carried out to check the validity of the assumptions. 
'''
# A test to determine whether there is a significant difference between 2 variables.

# Data:
#  Data shows the diameter of two units of cutlet
# randomly selected sample of cutlets was collected from both units and measured
# External Conditions are conducted in a controlled environment to ensure external conditions are same

# Business Problem: 
# Determine whether there is any significant difference in the diameter of the cutlet between two units

cutlet = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/16. Hypothesis Testing/Datasets_Hypothesis Testing/Cutlets.csv")

cutlet.columns = 'Unit_A', 'Unit_B'

# Normality test 
# Ho = Data are Normal
# Ha = Data are not Normal

print(stats.shapiro(cutlet.Unit_A[0:35]))  # p high null fly
print(stats.shapiro(cutlet.Unit_B[0:35]))  # p high null fly

# Data are normal
# Parametric Test case
# Assuming the external Conditions are same for both the samples
# Paired T-Test
# Ho: Diameter of the cutlet in Unit_A = Diameter of the cutlet in Unit_B
# Ha: Diameter of the cutlet in Unit_A != Diameter of the cutlet in Unit_B

ttest, pval = stats.ttest_ind(cutlet.Unit_A[0:35], cutlet.Unit_B[0:35])
print(pval)

# p-value = 0.47 > 0.05 => p high null fly
# Ho: Diameter of the cutlet in Unit_A = Diameter of the cutlet in Unit_B

# Conclusion: Diameter of the cutlet in both units are looking same 

'''2. A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. 
They collected a random sample and recorded TAT for reports from 4 laboratories. 
TAT is defined as a sample collected to report dispatch. 
Analyze the data and determine whether there is any difference in average TAT among the different laboratories at a 5% significance level
'''
# Business Problem: 
# Determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories.

lab_tat = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/16. Hypothesis Testing/Datasets_Hypothesis Testing/lab_tat_updated.csv")
lab_tat
lab_tat.describe()

# Normality Test - # Shapiro Test
# H0 = Data are Normal
# Ha = Data are not Normal
stats.shapiro(lab_tat.Laboratory_1) # pvalue 0.42 > 0.05 h0 fly
stats.shapiro(lab_tat.Laboratory_2) # pvalue 0.86 > 0.05 h0 fly
stats.shapiro(lab_tat.Laboratory_3) # pvalue 0.06 > 0.05 h0 fly
stats.shapiro(lab_tat.Laboratory_4) # pvalue 0.66 > 0.05 h0 fly

# Variance test
# Ho: All the 4 Laboratories have equal average Turn Around Time (TAT)
# Ha: All the 4 Laboratories have unequal average Turn Around Time (TAT)
scipy.stats.levene(lab_tat.Laboratory_1, lab_tat.Laboratory_2, lab_tat.Laboratory_3, lab_tat.Laboratory_4)
# Variances are statisticaly equal; pvalue 0.38 > 0.05 h0 fly

# One - Way Anova(ANalysis Of VAriance)
# Ho: All the 4 Laboratories have equal average Turn Around Time (TAT)
# Ha: All the 4 Laboratories have unequal average Turn Around Time (TAT)

F, p = stats.f_oneway(lab_tat.Laboratory_1, lab_tat.Laboratory_2, lab_tat.Laboratory_3, lab_tat.Laboratory_4)
p 

# P Low Null go
# Conclusion: There is a significant difference in the average Turn Around Time among the laboratories.

'''3. Sales of products in four different regions are tabulated for males and females. 
Find if male-female buyer rations are similar across regions.
'''
# Business Problem:
# Ensuring consistency in the male-female buyer ratios across different regions. 
buyer_ratio = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/16. Hypothesis Testing/Datasets_Hypothesis Testing/BuyerRatio.csv")

# Observed values
observed_values = np.array([[50, 142, 131, 70],   # Males
    [435, 1523, 1356, 750]  # Females
])

# Ho: All regions have equal proportions of male & female buyers
# Ha: Not all cregions have equal proportions of male & female buyers
Chisquares_results = scipy.stats.chi2_contingency(observed_values)
print(Chisquares_results)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

# p-value = 0.66 > 0.05 => P high Null fly

# All regions have equal proportions of males & females
# Conclusion: All Proportions are equal 

'''4. Telecall uses 4 centers around the globe to process customer order forms. 
They audit a certain % of the customer order forms. Any error in the order form renders it defective and must be reworked before processing. 
The manager wants to check whether the defective % varies by center.
Please analyze the data at a 5% significance level and help the manager draw appropriate inferences.
'''
order_form = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/16. Hypothesis Testing/Datasets_Hypothesis Testing/CustomerOrderform.csv")
order_form

# Convert categorical values to numeric counts
df = pd.get_dummies(order_form[0:300]).groupby(level=0, axis=1).sum()

# Perform chi-square test for independence
chi2, p, dof, expected = scipy.stats.chi2_contingency(df)

# Print the test results
print(f"Chi-Square Statistic: {chi2}")
print(f"P-Value: {p}")

# p-value = 1.0 > 0.05 => P high Null fly

# Defective percentages do not significantly differ among the centers, indicating relatively consistent performance across all centers
# Conclusion: All Proportions are equal 

'''5. Fantaloons Sales managers commented that % of males versus females walking into the store differs based on the day of the week. 
Analyze the data and determine whether there is evidence at a 5 % significance level to support this hypothesis. 
'''
# Business Problem:
# To analyze the difference in the percentage of males versus females walking into the store based on the day of the week.

fant = pd.read_csv("C:/Users/RamyaRajaLakshmi/Documents/My_Course/Machine Learning/16. Hypothesis Testing/Datasets_Hypothesis Testing/Fantaloons.csv")
df = fant[0:400]

tab1 = df.Weekdays.value_counts()
print(tab1)

tab2 = df.Weekend.value_counts()
print(tab2)

# Generate crosstab of 'Male' and 'Female' occurrences in 'Weekdays' and 'Weekend'
crosstab = pd.crosstab(df['Weekdays'], df['Weekend'])
print(crosstab)

# Calculate the total number of Female and Male across both Weekdays and Weekend
total_female = crosstab['Female'].sum()
total_male = crosstab['Male'].sum()

count = np.array([167, 47]) # Female & Male occurences on both weekdays & weekend
nobs = np.array([total_female, total_male]) # Total no of female & male 

# Case1: Two Sided test
# Ho: Proportions of Female = Proportions of Male
# Ha: Proportions of Female != Proportions of Male

stats, pval = proportions_ztest(count, nobs, alternative = 'two-sided') 
print("%.2f" % pval)
# P-value = 0.000 < 0.05 => P low Null go
# Ha: Proportions of Female != Proportions of Male

# Case2: One-sided (Greater) test
# Ho: Proportions of Female <= Proportions of Male
# Ha: Proportions of Female > Proportions of Male
stats, pval = proportions_ztest(count, nobs, alternative = 'larger')
print("%.2f" % pval) 
# P-value = 0.0 > 0.05 => P low Null go

# Ho: Proportions of Female > Proportions of Male
# Conclusion: The proportion of 'Female' occurrences is greater than the proportion of 'Male' occurrences.
