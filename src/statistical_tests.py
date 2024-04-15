import pandas as pd 
import numpy as np 
from scipy.stats import anderson, ttest_rel, chi2_contingency, wilcoxon
from src.visualizations import name_to_str 
from src.fairness import compute_RDP, compute_PR, compute_UCPR
import matplotlib.pyplot as plt
import seaborn as sns 

# Statistically Testing Fairness. The metric can be "rdp", "pr", or "ucpr" 
def testing_fairness(dfs, setting, methods, races, alpha=0.05, metric="rdp", n=1400):
    if metric == "rdp":
        observed_freq = compute_RDP(dfs, setting, methods, races)
    elif metric == "pr":
        observed_freq = compute_PR(dfs, setting, methods, races)
    elif metric == "ucpr":
        observed_freq = compute_UCPR(dfs, setting, methods, races)
    
    chi2_results = [] 
    for method in methods:
        
        observed_counts = [int(n*freq) for freq in observed_freq[method]]
        
      
        expected_counts = [1 / len(observed_counts) * n] * len(observed_counts)  
     
        # Perform Pearson's chi-squared test
        chi2, p_value, dof, expected = chi2_contingency([observed_counts, expected_counts])

        result = {
            'Method': method, 
            'Statistic': chi2,
            'P-Value': p_value
        }
        
        # Append the result to the list
        chi2_results.append(result)
        
    # Create a DataFrame from the list of results
    results_df = pd.DataFrame(chi2_results)
    results_df["Reject"] = results_df["P-Value"] < alpha

    return results_df 

### Two Sample Tests: 
def two_sample_wilcoxon(dfs, alpha, methods, losses, return_decision=True):
    decisions = []
    for method in methods:
        # List to store t-test results
        wilcoxon_results = []

        for col in losses:
            if col == "race_0-1":
                continue
            # Perform t-test for each column in df1 and df2
            df_fairface = dfs["fairface"][method]
            df_unfairface = dfs["unfairface"][method] 
            statistic, p_value = wilcoxon(df_fairface[col], 
                                             df_unfairface[col])
            
            # Store results in a dictionary
            result = {
                'Column': col,
                'Statistic': statistic,
                'P-Value': p_value
            }
            
            # Append the result to the list
            wilcoxon_results.append(result)

        # Create a DataFrame from the list of results
        results_df = pd.DataFrame(wilcoxon_results)
        results_df["Reject"] = results_df["P-Value"] < alpha

        if return_decision:
            decision = {"method": name_to_str(method)}
            for loss in losses:
                if loss == "race_0-1":
                    continue
                
                decision[loss] = results_df["P-Value"][results_df["Column"]==loss].item() < alpha
            decisions.append(decision)
        else:
            # only return p-values 
            decision = {"method": name_to_str(method)}
            for loss in losses:
                if loss == "race_0-1":
                    continue
                
                decision[loss] = results_df["P-Value"][results_df["Column"]==loss].item()
            decisions.append(decision)
            
    decisions_df = pd.concat([pd.DataFrame([decision]) for decision in decisions], ignore_index=True)
    return decisions_df

def test_paired_t_test_assumption(dfs, methods, losses, plot=False):
    for method in methods:
        for col in losses:
            if col == "race_0-1":
                continue 
            df_fairface = dfs["fairface"][method]
            df_unfairface = dfs["unfairface"][method]
            dif = df_fairface[col] - df_unfairface[col]
            
            if plot:
                # plot the difference
                sns.histplot(dif)
                plt.show()
        
            # test for normality using Anderson-Darling
            result = anderson(dif, dist="norm")
            for i in range(len(result.significance_level)):
                if result.statistic < result.critical_values[i]:
                    print(f"Method = {method}, Loss = {col}")
                    print(f"At {result.significance_level[i]}% significance level, data looks normal (fail to reject H0)")
                 
def two_sample_paired_ttest(dfs, alpha, methods, losses, return_decision=True):
    decisions = []
    for method in methods:
        # List to store t-test results
        t_test_results = []

        for col in losses:
            if col == "race_0-1":
                continue 
            # Perform t-test for each column in df1 and df2
            df_fairface = dfs["fairface"][method]
            df_unfairface = dfs["unfairface"][method] 
            t_statistic, p_value = ttest_rel(df_fairface[col], 
                                             df_unfairface[col])
            
            # Store results in a dictionary
            result = {
                'Column': col,
                'T-Statistic': t_statistic,
                'P-Value': p_value
            }
            
            # Append the result to the list
            t_test_results.append(result)

        print(t_test_results)
        # Create a DataFrame from the list of results
        results_df = pd.DataFrame(t_test_results)
        results_df["Reject"] = results_df["P-Value"] < alpha
        
        decision = {"method": name_to_str(method)}
        for loss in losses:
            if loss == "race_0-1":
                continue
            if return_decision:
                decision[loss] = results_df["P-Value"][results_df["Column"]==loss].item() < alpha
            else:
                # return P-values 
                decision[loss] = results_df["P-Value"][results_df["Column"]==loss].item()
        decisions.append(decision)
    decisions_df = pd.concat([pd.DataFrame([decision]) for decision in decisions], ignore_index=True)
    return decisions_df

def two_sample_chi2(dfs, alpha, methods):
    decisions = []
    for method in methods:
        # List to store t-test results
        chi2_results = []
        col = "race_0-1"
        
        df_fairface = dfs["fairface"][method]
        df_unfairface = dfs["unfairface"][method] 
        n = len(df_fairface)
        ones = df_fairface[col].sum()
        counts_fairface = [ones, n - ones]
        ones = df_unfairface[col].sum()
        counts_unfairface = [ones, n - ones]
        statistic, p_value, dof, expected = chi2_contingency(np.array([counts_fairface, counts_unfairface]))
        
        # Store results in a dictionary
        result = {
            'Column': col,
            'Statistic': statistic,
            'P-Value': p_value
        }
        
        # Append the result to the list
        chi2_results.append(result)

        # Create a DataFrame from the list of results
        results_df = pd.DataFrame(chi2_results)
        results_df["Reject"] = results_df["P-Value"] < alpha
     
        decision = {"method": method}
        
        decision[col] = results_df["P-Value"][results_df["Column"]==col].item() < alpha
        decision[f"{col}-P-value"] = results_df["P-Value"][results_df["Column"]==col].item()
        decisions.append(decision)
    decisions_df = pd.concat([pd.DataFrame([decision]) for decision in decisions], ignore_index=True)
    return decisions_df