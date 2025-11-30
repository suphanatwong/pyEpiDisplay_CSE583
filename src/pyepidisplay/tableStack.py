"""
tableStack: Tabulation of variables in a stack form

This module provides functionality for tabulating variables with the same possible 
range of distribution and stacking them into a new table with descriptive statistics 
or breakdown distribution against a column variable.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, f_oneway, ttest_ind, bartlett, shapiro
from sklearn.decomposition import FactorAnalysis
import warnings


class TableStackResult:
    """Container for tableStack results"""
    
    def __init__(self, results, items_reversed=None, item_labels=None, 
                 total_score=None, mean_score=None, stats_dict=None):
        self.results = results
        self.items_reversed = items_reversed
        self.item_labels = item_labels
        self.total_score = total_score
        self.mean_score = mean_score
        if stats_dict:
            self.mean_of_total_scores = stats_dict.get('mean_of_total_scores')
            self.sd_of_total_scores = stats_dict.get('sd_of_total_scores')
            self.mean_of_average_scores = stats_dict.get('mean_of_average_scores')
            self.sd_of_average_scores = stats_dict.get('sd_of_average_scores')
    
    def __repr__(self):
        if isinstance(self.results, pd.DataFrame):
            return str(self.results)
        return str(self.results)


def tablestack(vars, dataFrame, minlevel="auto", maxlevel="auto", count=True, 
                na_rm=False, means=True, medians=False, sds=True, decimal=1,
                total=True, var_labels=True, var_labels_trunc=150, 
                reverse=False, vars_to_reverse=None, by=None, vars_to_factor=None,
                iqr="auto", prevalence=False, percent="column", frequency=True,
                test=True, name_test=True, total_column=False, 
                simulate_p_value=False, sample_size=True, assumption_p_value=0.01):
    """
    Tabulation of variables in a stack form
    
    Parameters
    ----------
    vars : list or range
        Vector of column indices or names in the data frame
    dataFrame : pd.DataFrame
        Source data frame of the variables
    minlevel : str or numeric
        Possible minimum value of items (default: "auto")
    maxlevel : str or numeric
        Possible maximum value of items (default: "auto")
    count : bool
        Whether number of valid records for each item will be displayed
    na_rm : bool
        Whether missing values would be removed during calculation
    means : bool
        Whether means of all selected items will be displayed
    medians : bool
        Whether medians of all selected items will be displayed
    sds : bool
        Whether standard deviations of all selected items will be displayed
    decimal : int
        Number of decimals displayed in the statistics
    total : bool
        Display of means and standard deviations of total and average scores
    var_labels : bool
        Presence of descriptions of variables
    var_labels_trunc : int
        Number of characters used for variable description
    reverse : bool
        Whether items negatively correlated will be reversed
    vars_to_reverse : list
        Specific variables to reverse
    by : str or int
        A variable for column breakdown
    vars_to_factor : list
        Variables to be converted to categorical
    iqr : str or list
        Variables to display median and inter-quartile range
    prevalence : bool
        For dichotomous variables, show prevalence
    percent : str
        Type of percentage: "column", "row", or "none"
    frequency : bool
        Whether to display frequency in cells
    test : bool
        Whether statistical tests will be computed
    name_test : bool
        Display name of the test and degrees of freedom
    total_column : bool
        Whether to add 'total column' to output
    simulate_p_value : bool
        Simulate P value for Fisher's exact test
    sample_size : bool
        Whether to display sample size of each column
    assumption_p_value : float
        Level for Bartlett's test P value
    
    Returns
    -------
    TableStackResult
        Object containing results table and additional statistics
    """
    
    # Convert vars to list of column indices
    if isinstance(vars, range):
        selected = list(vars)
    elif isinstance(vars, (list, tuple)):
        selected = []
        for v in vars:
            if isinstance(v, str):
                if v in dataFrame.columns:
                    selected.append(dataFrame.columns.get_loc(v))
                else:
                    raise ValueError(f"Column '{v}' not found in dataFrame")
            else:
                selected.append(v)
    elif isinstance(vars, str):
        # Single string variable
        if vars in dataFrame.columns:
            selected = [dataFrame.columns.get_loc(vars)]
        else:
            raise ValueError(f"Column '{vars}' not found in dataFrame")
    else:
        selected = [vars]
    
    # Process by variable
    by_var = None
    by1 = None
    if by is not None:
        if isinstance(by, list):
            # If by is a list with one element, extract it
            if len(by) == 1:
                by = by[0]
            else:
                raise ValueError("'by' parameter should contain only one variable")
        
        if isinstance(by, str):
            if by in dataFrame.columns:
                by_var = dataFrame.columns.get_loc(by)
                by1 = dataFrame.iloc[:, by_var].astype('category')
            else:
                # Special case for "Total" column only
                by1 = pd.Categorical(['Total'] * len(dataFrame))
        elif isinstance(by, int):
            by_var = by
            by1 = dataFrame.iloc[:, by_var].astype('category')
    
    # Process vars_to_factor
    selected_to_factor = []
    if vars_to_factor is not None:
        if isinstance(vars_to_factor, (list, tuple)):
            for v in vars_to_factor:
                if isinstance(v, str):
                    selected_to_factor.append(dataFrame.columns.get_loc(v))
                else:
                    selected_to_factor.append(v)
        else:
            selected_to_factor = [vars_to_factor]
    
    # Process iqr selection
    selected_iqr = []
    if isinstance(iqr, str):
        if iqr == "auto":
            selected_iqr = "auto"
        else:
            selected_iqr = None
    elif isinstance(iqr, (list, tuple)):
        for v in iqr:
            if isinstance(v, str):
                selected_iqr.append(dataFrame.columns.get_loc(v))
            else:
                selected_iqr.append(v)
    
    # Validate and convert selected variables
    selected_df = dataFrame.iloc[:, selected].copy()
    
    # Convert numeric columns if needed
    for i in selected:
        if pd.api.types.is_numeric_dtype(dataFrame.iloc[:, i]) and by is not None:
            if i in selected_to_factor:
                dataFrame.iloc[:, i] = dataFrame.iloc[:, i].astype('category')
            else:
                dataFrame.iloc[:, i] = pd.to_numeric(dataFrame.iloc[:, i], errors='coerce')
    
    # Check for reverse on factors
    if (reverse or (vars_to_reverse is not None and len(vars_to_reverse) > 0)):
        if pd.api.types.is_categorical_dtype(selected_df.iloc[:, 0]):
            raise ValueError("Variables must be numeric before reversing")
    
    # NO BY VARIABLE - Simple stacking
    if by is None:
        return _table_stack_no_by(
            selected, dataFrame, selected_df, minlevel, maxlevel, 
            count, means, medians, sds, decimal, total, var_labels,
            var_labels_trunc, reverse, vars_to_reverse
        )
    
    # WITH BY VARIABLE - Breakdown analysis
    else:
        return _table_stack_with_by(
            selected, dataFrame, by1, selected_iqr, selected_to_factor,
            decimal, var_labels, prevalence, percent, frequency,
            test, name_test, total_column, simulate_p_value, 
            sample_size, assumption_p_value
        )


def _table_stack_no_by(selected, dataFrame, selected_df, minlevel, maxlevel,
                       count, means, medians, sds, decimal, total, 
                       var_labels, var_labels_trunc, reverse, vars_to_reverse):
    """Handle tableStack without by variable"""
    
    # Create numeric matrix
    selected_matrix = selected_df.apply(pd.to_numeric, errors='coerce').values
    
    # Determine min/max levels
    if minlevel == "auto":
        minlevel = int(np.nanmin(selected_matrix))
    if maxlevel == "auto":
        maxlevel = int(np.nanmax(selected_matrix))
    
    nlevel = list(range(minlevel, maxlevel + 1))
    
    # Handle variable reversal
    sign1 = np.ones(len(selected))
    
    if vars_to_reverse is not None and len(vars_to_reverse) > 0:
        which_neg = []
        for v in vars_to_reverse:
            if isinstance(v, str):
                which_neg.append(dataFrame.columns.get_loc(v))
            else:
                which_neg.append(v)
        
        for idx, i in enumerate(selected):
            if i in which_neg:
                selected_matrix[:, idx] = maxlevel + 1 - selected_matrix[:, idx]
                sign1[idx] = -1
        reverse = False
    
    elif reverse:
        # Check for highly correlated variables
        valid_data = selected_matrix[~np.isnan(selected_matrix).any(axis=1)]
        if len(valid_data) > 1:
            matR1 = np.corrcoef(valid_data.T)
            np.fill_diagonal(matR1, 0)
            
            if np.any(matR1 > 0.98):
                reverse = False
                warnings.warn("Extremely correlated variables detected. Reverse disabled.")
            else:
                # Perform factor analysis for reversal
                try:
                    fa = FactorAnalysis(n_components=1, random_state=0)
                    scores = fa.fit_transform(valid_data)
                    
                    for idx in range(len(selected)):
                        corr = np.corrcoef(scores[:, 0], valid_data[:, idx])[0, 1]
                        sign1[idx] = np.sign(corr)
                        if sign1[idx] < 0:
                            selected_matrix[:, idx] = maxlevel + minlevel - selected_matrix[:, idx]
                except:
                    warnings.warn("Factor analysis failed. Reverse disabled.")
                    reverse = False
    
    # Build table
    table_data = []
    
    for idx, i in enumerate(selected):
        col_data = dataFrame.iloc[:, i]
        
        # Create frequency table
        if not pd.api.types.is_categorical_dtype(col_data) and not pd.api.types.is_bool_dtype(col_data):
            x = pd.Categorical(col_data, categories=nlevel)
            tablei = x.value_counts().reindex(nlevel, fill_value=0).values
        elif pd.api.types.is_bool_dtype(col_data):
            tablei = col_data.value_counts().reindex([False, True], fill_value=0).values
        else:
            tablei = col_data.value_counts().values
        
        row_data = list(tablei)
        
        # Add count
        if count:
            row_data.append(col_data.notna().sum())
        
        # Add statistics for numeric/boolean
        if pd.api.types.is_numeric_dtype(col_data) or pd.api.types.is_bool_dtype(col_data):
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            
            if means:
                row_data.append(round(numeric_data.mean(), decimal))
            if medians:
                row_data.append(round(numeric_data.median(), decimal))
            if sds:
                row_data.append(round(numeric_data.std(), decimal))
        
        table_data.append(row_data)
    
    # Create DataFrame
    col_names = [str(x) for x in nlevel]
    if count:
        col_names.append('count')
    if means and (pd.api.types.is_numeric_dtype(selected_df.iloc[:, 0]) or 
                  pd.api.types.is_bool_dtype(selected_df.iloc[:, 0])):
        col_names.append('mean')
    if medians and (pd.api.types.is_numeric_dtype(selected_df.iloc[:, 0]) or 
                    pd.api.types.is_bool_dtype(selected_df.iloc[:, 0])):
        col_names.append('median')
    if sds and (pd.api.types.is_numeric_dtype(selected_df.iloc[:, 0]) or 
                pd.api.types.is_bool_dtype(selected_df.iloc[:, 0])):
        col_names.append('sd')
    
    results = pd.DataFrame(table_data, columns=col_names)
    
    # Set row names
    if var_labels:
        results.index = [dataFrame.columns[i] for i in selected]
    else:
        results.index = [f"{i}: {dataFrame.columns[i]}" for i in selected]
    
    # Add total scores if requested
    stats_dict = {}
    total_score = None
    mean_score = None
    
    if total and (pd.api.types.is_numeric_dtype(selected_df.iloc[:, 0]) or 
                  pd.api.types.is_bool_dtype(selected_df.iloc[:, 0])):
        total_score = np.nansum(selected_matrix, axis=1)
        mean_score = np.nanmean(selected_matrix, axis=1)
        
        mean_of_total = np.nanmean(total_score)
        sd_of_total = np.nanstd(total_score, ddof=1)
        mean_of_average = np.nanmean(mean_score)
        sd_of_average = np.nanstd(mean_score, ddof=1)
        
        stats_dict = {
            'mean_of_total_scores': mean_of_total,
            'sd_of_total_scores': sd_of_total,
            'mean_of_average_scores': mean_of_average,
            'sd_of_average_scores': sd_of_average
        }
        
        # Add total rows
        total_row = [''] * len(col_names)
        total_row[col_names.index('count')] = len(total_score[~np.isnan(total_score)])
        if 'mean' in col_names:
            total_row[col_names.index('mean')] = round(mean_of_total, decimal)
        if 'sd' in col_names:
            total_row[col_names.index('sd')] = round(sd_of_total, decimal)
        
        avg_row = [''] * len(col_names)
        avg_row[col_names.index('count')] = len(mean_score[~np.isnan(mean_score)])
        if 'mean' in col_names:
            avg_row[col_names.index('mean')] = round(mean_of_average, decimal)
        if 'sd' in col_names:
            avg_row[col_names.index('sd')] = round(sd_of_average, decimal)
        
        total_df = pd.DataFrame([total_row, avg_row], 
                               columns=col_names,
                               index=[' Total score', ' Average score'])
        results = pd.concat([results, total_df])
    
    # Identify reversed items
    items_reversed = None
    if reverse or (vars_to_reverse is not None):
        items_reversed = [dataFrame.columns[selected[i]] for i in range(len(selected)) if sign1[i] < 0]
    
    return TableStackResult(
        results=results,
        items_reversed=items_reversed,
        total_score=total_score,
        mean_score=mean_score,
        stats_dict=stats_dict
    )


def _table_stack_with_by(selected, dataFrame, by1, selected_iqr, selected_to_factor,
                         decimal, var_labels, prevalence, percent, frequency,
                         test, name_test, total_column, simulate_p_value,
                         sample_size, assumption_p_value):
    """Handle tableStack with by variable"""
    
    # Validate by1
    if by1 is None:
        raise ValueError("by1 cannot be None in _table_stack_with_by")
    
    if not isinstance(by1, pd.Categorical):
        by1 = pd.Categorical(by1)
    
    # Determine which variables need IQR
    if selected_iqr == "auto":
        selected_iqr = []
        for i in selected:
            col = dataFrame.iloc[:, i]
            if pd.api.types.is_numeric_dtype(col):
                if len(by1.categories) > 1:
                    try:
                        # Test for normality and homogeneity
                        groups = [col[by1 == cat].dropna() for cat in by1.categories]
                        groups = [g for g in groups if len(g) >= 3]
                        
                        if len(groups) >= 2:
                            if len(col) < 5000:
                                # Shapiro test on residuals
                                residuals = col - col.groupby(by1).transform('mean')
                                if len(residuals.dropna()) >= 3:
                                    _, p_shapiro = shapiro(residuals.dropna())
                                else:
                                    p_shapiro = 1.0
                            else:
                                sampled = np.random.choice(col.dropna(), min(250, len(col.dropna())), replace=False)
                                _, p_shapiro = shapiro(sampled)
                            
                            # Bartlett test
                            _, p_bartlett = bartlett(*groups)
                            
                            if p_shapiro < assumption_p_value or p_bartlett < assumption_p_value:
                                selected_iqr.append(i)
                    except:
                        pass
    elif selected_iqr is None:
        selected_iqr = []
    
    # Check if only one level
    if len(by1.categories) == 1:
        test = False
    name_test = name_test if test else False
    
    # Build table data as dictionary for proper DataFrame construction
    table_data = []
    row_labels = []
    
    # Prepare column structure
    n_by_cols = len(by1.categories)
    
    # Add sample size row
    if sample_size:
        sample_counts = [by1.value_counts().get(cat, 0) for cat in by1.categories]
        
        sample_row = {}
        for idx, cat in enumerate(by1.categories):
            sample_row[str(cat)] = sample_counts[idx]
        
        if total_column:
            sample_row['Total'] = len(by1)
        if test:
            if name_test:
                sample_row['Test'] = ''
                sample_row['P-value'] = ''
            else:
                sample_row['P-value'] = ''
        
        table_data.append(sample_row)
        row_labels.append('N')
    
    # Process each variable
    for i in selected:
        col = dataFrame.iloc[:, i]
        var_name = dataFrame.columns[i] if var_labels else f"{i}: {dataFrame.columns[i]}"
        
        # Categorical/Factor variable
        if pd.api.types.is_categorical_dtype(col) or pd.api.types.is_bool_dtype(col) or i in selected_to_factor:
            if not pd.api.types.is_categorical_dtype(col):
                col = col.astype('category')
            
            # Create contingency table
            ct = pd.crosstab(col, by1)
            
            # Check for zero counts
            if (ct == 0).any().any():
                warnings.warn(f"Variable {dataFrame.columns[i]} has zero count in at least one cell")
            
            # Perform test first to get p-value for header
            p_value = None
            test_method = ''
            if test:
                ct_test = ct.copy()
                expected = np.outer(ct_test.sum(axis=1), ct_test.sum(axis=0)) / ct_test.sum().sum()
                
                if (expected < 5).sum() / expected.size > 0.2 and len(dataFrame) < 1000:
                    test_method = "Fisher's exact"
                    if ct_test.shape == (2, 2):
                        _, p_value = fisher_exact(ct_test)
                    else:
                        p_value = np.nan
                else:
                    chi2, p_value, dof, _ = chi2_contingency(ct_test, correction=False)
                    test_method = f"Chi-sq({dof}df)={round(chi2, decimal+1)}"
            
            # Format table
            if len(ct) == 2 and prevalence:
                # Show prevalence for dichotomous
                prev_data = {}
                for cat in by1.categories:
                    n_positive = ct.loc[ct.index[1], cat]
                    n_total = ct[cat].sum()
                    pct = round(n_positive / n_total * 100, decimal) if n_total > 0 else 0
                    prev_data[str(cat)] = f"{n_positive}/{n_total} ({pct}%)"
                
                if total_column:
                    ct_total = pd.crosstab(col, pd.Categorical(['Total'] * len(col)))
                    n_positive = ct_total.loc[ct_total.index[1], 'Total']
                    n_total = ct_total['Total'].sum()
                    pct = round(n_positive / n_total * 100, decimal) if n_total > 0 else 0
                    prev_data['Total'] = f"{n_positive}/{n_total} ({pct}%)"
                
                if test:
                    if name_test:
                        prev_data['Test'] = test_method
                        prev_data['P-value'] = f"< 0.001" if p_value < 0.001 else round(p_value, decimal + 2)
                    else:
                        prev_data['P-value'] = f"< 0.001" if p_value < 0.001 else round(p_value, decimal + 2)
                
                table_data.append(prev_data)
                row_labels.append(f"{var_name} = {ct.index[1]}")
            else:
                # Add variable header row with test results
                header_data = {}
                for cat in by1.categories:
                    header_data[str(cat)] = ''
                if total_column:
                    header_data['Total'] = ''
                if test:
                    if name_test:
                        header_data['Test'] = test_method
                        header_data['P-value'] = f"< 0.001" if p_value < 0.001 else round(p_value, decimal + 2)
                    else:
                        header_data['P-value'] = f"< 0.001" if p_value < 0.001 else round(p_value, decimal + 2)
                
                table_data.append(header_data)
                row_labels.append(var_name)
                
                # Regular cross-tabulation
                for level in ct.index:
                    level_data = {}
                    for cat in by1.categories:
                        count = ct.loc[level, cat]
                        if percent == "column":
                            pct = round(count / ct[cat].sum() * 100, decimal) if ct[cat].sum() > 0 else 0
                        elif percent == "row":
                            pct = round(count / ct.loc[level].sum() * 100, decimal) if ct.loc[level].sum() > 0 else 0
                        else:
                            pct = None
                        
                        if frequency and pct is not None:
                            level_data[str(cat)] = f"{count} ({pct}%)"
                        elif pct is not None:
                            level_data[str(cat)] = f"{pct}%"
                        else:
                            level_data[str(cat)] = str(count)
                    
                    if total_column:
                        ct_total = pd.crosstab(col, pd.Categorical(['Total'] * len(col)))
                        count = ct_total.loc[level, 'Total']
                        level_data['Total'] = str(count)
                    
                    if test:
                        if name_test:
                            level_data['Test'] = ''
                            level_data['P-value'] = ''
                        else:
                            level_data['P-value'] = ''
                    
                    table_data.append(level_data)
                    row_labels.append(f"  {level}")
        
        # Numeric variable
        elif pd.api.types.is_numeric_dtype(col):
            # Perform test first
            p_value = None
            test_method = ''
            if test:
                groups = [col[by1 == cat].dropna() for cat in by1.categories]
                if any(len(g) < 3 for g in groups):
                    test_method = "Sample too small"
                    p_value = np.nan
                else:
                    if i in selected_iqr:
                        if len(groups) > 2:
                            test_method = "Kruskal-Wallis test"
                            _, p_value = kruskal(*groups)
                        else:
                            test_method = "Mann-Whitney test"
                            _, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    else:
                        if len(groups) > 2:
                            f_stat, p_value = f_oneway(*groups)
                            dof1 = len(groups) - 1
                            dof2 = len(col.dropna()) - len(groups)
                            test_method = f"ANOVA F({dof1},{dof2}df)={round(f_stat, decimal+1)}"
                        else:
                            t_stat, p_value = ttest_ind(groups[0], groups[1], equal_var=True)
                            dof = len(col.dropna()) - 2
                            test_method = f"t-test({dof}df)={round(abs(t_stat), decimal+1)}"
            
            # Add variable header with test
            header_data = {}
            for cat in by1.categories:
                header_data[str(cat)] = ''
            if total_column:
                header_data['Total'] = ''
            if test:
                if name_test:
                    header_data['Test'] = test_method
                    header_data['P-value'] = f"< 0.001" if p_value < 0.001 else round(p_value, decimal + 2) if p_value is not None else 'NA'
                else:
                    header_data['P-value'] = f"< 0.001" if p_value < 0.001 else round(p_value, decimal + 2) if p_value is not None else 'NA'
            
            table_data.append(header_data)
            row_labels.append(var_name)
            
            # Add statistics row
            stats_data = {}
            if i in selected_iqr:
                # Use median and IQR
                for cat in by1.categories:
                    data = col[by1 == cat].dropna()
                    if len(data) > 0:
                        q1, median, q3 = data.quantile([0.25, 0.5, 0.75])
                        stats_data[str(cat)] = f"{round(median, decimal)} ({round(q1, decimal)}, {round(q3, decimal)})"
                    else:
                        stats_data[str(cat)] = "NA"
                
                if total_column:
                    q1, median, q3 = col.quantile([0.25, 0.5, 0.75])
                    stats_data['Total'] = f"{round(median, decimal)} ({round(q1, decimal)}, {round(q3, decimal)})"
                
                if test:
                    if name_test:
                        stats_data['Test'] = ''
                        stats_data['P-value'] = ''
                    else:
                        stats_data['P-value'] = ''
                
                table_data.append(stats_data)
                row_labels.append("  Median (IQR)")
            else:
                # Use mean and SD
                for cat in by1.categories:
                    data = col[by1 == cat].dropna()
                    if len(data) > 0:
                        mean_val = round(data.mean(), decimal)
                        sd_val = round(data.std(), decimal)
                        stats_data[str(cat)] = f"{mean_val} ({sd_val})"
                    else:
                        stats_data[str(cat)] = "NA"
                
                if total_column:
                    mean_val = round(col.mean(), decimal)
                    sd_val = round(col.std(), decimal)
                    stats_data['Total'] = f"{mean_val} ({sd_val})"
                
                if test:
                    if name_test:
                        stats_data['Test'] = ''
                        stats_data['P-value'] = ''
                    else:
                        stats_data['P-value'] = ''
                
                table_data.append(stats_data)
                row_labels.append("  Mean (SD)")
    
    # Create DataFrame with proper structure
    results = pd.DataFrame(table_data, index=row_labels)
    
    return TableStackResult(results=results)
    #     selected_iqr = []
    #     for i in selected:
    #         col = dataFrame.iloc[:, i]
    #         if pd.api.types.is_numeric_dtype(col):
    # return TableStackResult(results=results)


# # Example usage and testing
# if __name__ == "__main__":
#     # Create sample data
#     np.random.seed(42)
#     df = pd.DataFrame({
#         'age': np.random.randint(20, 70, 100),
#         'score1': np.random.randint(1, 6, 100),
#         'score2': np.random.randint(1, 6, 100),
#         'score3': np.random.randint(1, 6, 100),
#         'group': np.random.choice(['A', 'B', 'C'], 100)
#     })
    
#     # Test without by variable
#     result1 = table_stack(vars=range(1, 4), dataFrame=df, 
#                          means=True, sds=True, total=True)
#     print("Without by variable:")
#     print(result1)
#     print()
    
#     # Test with by variable
#     result2 = table_stack(vars=range(1, 4), dataFrame=df, 
#                          by='group', test=True)
#     print("With by variable:")
#     print(result2)

# import numpy as np
# import pandas as pd
# from scipy import stats
# from typing import Union, List, Optional, Dict, Any
# import warnings


# class TableStack:
    
#     def __init__(self, data: pd.DataFrame):
#         self.data = data.copy()
#         self.var_labels = {}
        
#     def set_var_labels(self, labels: Dict[str, str]):
#         self.var_labels = labels
        
#     def tablestack(self,
#                    vars: List[Union[str, int]],
#                    minlevel: Union[str, int] = "auto",
#                    maxlevel: Union[str, int] = "auto",
#                    count: bool = True,
#                    na_rm: bool = False,
#                    means: bool = True,
#                    medians: bool = False,
#                    sds: bool = True,
#                    decimal: int = 1,
#                    total: bool = True,
#                    var_labels: bool = True,
#                    var_labels_trunc: int = 150,
#                    reverse: bool = False,
#                    vars_to_reverse: Optional[List[Union[str, int]]] = None,
#                    by: Optional[Union[str, int]] = None,
#                    vars_to_factor: Optional[List[Union[str, int]]] = None,
#                    iqr: Union[str, List[Union[str, int]]] = "auto",
#                    prevalence: bool = False,
#                    percent: str = "column",
#                    frequency: bool = True,
#                    test: bool = True,
#                    name_test: bool = True,
#                    total_column: bool = False,
#                    simulate_p_value: bool = False,
#                    sample_size: bool = True,
#                    assumption_p_value: float = 0.01) -> Union[pd.DataFrame, Dict[str, Any]]:

#         # Normalize vars into a flat list
#         if isinstance(vars, (tuple, np.ndarray, pd.Index)):
#             vars = list(vars)
#         elif isinstance(vars, list):
#             # flatten nested list if needed
#             vars = [item for sub in vars for item in (sub if isinstance(sub, (list, tuple)) else [sub])]
#         else:
#             vars = [vars]

#         # Convert column indices to names if needed
#         selected_vars = []
#         for v in vars:
#             if isinstance(v, int):
#                 if v < 0 or v >= len(self.data.columns):
#                     raise IndexError(f"Column index {v} is out of range.")
#                 selected_vars.append(self.data.columns[v])
#             else:
#                 selected_vars.append(v)

#         # Ensure all selected columns exist
#         missing = [v for v in selected_vars if v not in self.data.columns]
#         if missing:
#             raise KeyError(f"Variables not found in data: {missing}")

#         # Handle vars_to_factor
#         if vars_to_factor is not None:
#             vars_to_factor_names = []
#             for v in vars_to_factor:
#                 if isinstance(v, int):
#                     vars_to_factor_names.append(self.data.columns[v])
#                 else:
#                     vars_to_factor_names.append(v)

#             for var in vars_to_factor_names:
#                 if var in selected_vars:
#                     self.data[var] = self.data[var].astype('category')

#         # Extract selected data
#         selected_df = self.data[selected_vars].copy()

#         # Case 1: No grouping variable
#         if by is None:
#             return self._table_stack_ungrouped(
#                 selected_df, selected_vars, minlevel, maxlevel, count,
#                 na_rm, means, medians, sds, decimal, total, var_labels,
#                 var_labels_trunc, reverse, vars_to_reverse
#             )

#         # Case 2: With grouping variable (not fully implemented here)
#         else:
#             if isinstance(by, int):
#                 by_var = self.data.columns[by]
#             else:
#                 by_var = by
#             return self._table_stack_grouped(
#                 selected_df, selected_vars, by_var, iqr, decimal,
#                 test, name_test, total_column, sample_size, prevalence,
#                 percent, frequency, var_labels, vars_to_factor,
#                 assumption_p_value, simulate_p_value
#             )
    
#     def _table_stack_ungrouped(self, selected_df, selected_vars, minlevel,
#                                maxlevel, count, na_rm, means, medians, sds,
#                                decimal, total, var_labels, var_labels_trunc,
#                                reverse, vars_to_reverse):
        
#         results = []
        
#         # Determine min/max levels
#         if minlevel == "auto":
#             minlevel = selected_df.select_dtypes(include=[np.number]).min().min()
#         if maxlevel == "auto":
#             maxlevel = selected_df.select_dtypes(include=[np.number]).max().max()
            
#         for var in selected_vars:
#             row_data = {}
#             col_data = selected_df[var]
            
#             # Handle categorical/factor variables
#             if pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):  #isinstance is too strict
#                 value_counts = col_data.value_counts()
#                 for level, cnt in value_counts.items():
#                     row_data[str(level)] = cnt
#             else:
#                 # Numeric variable - create levels
#                 if minlevel != "auto" and maxlevel != "auto":
#                     for level in range(int(minlevel), int(maxlevel) + 1):
#                         row_data[str(level)] = (col_data == level).sum()
            
#             # Add statistics
#             if count:
#                 row_data['count'] = col_data.notna().sum()
                
#             if pd.api.types.is_numeric_dtype(col_data):
#                 if means:
#                     row_data['mean'] = round(col_data.mean(), decimal)
#                 if medians:
#                     row_data['median'] = round(col_data.median(), decimal)
#                 if sds:
#                     row_data['sd'] = round(col_data.std(), decimal)
            
#             # Add variable label
#             if var_labels and var in self.var_labels:
#                 row_data['description'] = self.var_labels[var][:var_labels_trunc]
                
#             results.append(row_data)
        
#         # Create DataFrame
#         result_df = pd.DataFrame(results, index=selected_vars)
        
#         # Add total scores if numeric
#         if total and all(pd.api.types.is_numeric_dtype(selected_df[v]) for v in selected_vars):
#             total_scores = selected_df.sum(axis=1)
#             avg_scores = selected_df.mean(axis=1)
            
#             total_row = {
#                 'count': total_scores.notna().sum(),
#                 'mean': round(total_scores.mean(), decimal),
#                 'sd': round(total_scores.std(), decimal)
#             }
#             avg_row = {
#                 'count': avg_scores.notna().sum(),
#                 'mean': round(avg_scores.mean(), decimal),
#                 'sd': round(avg_scores.std(), decimal)
#             }
            
#             result_df = pd.concat([
#                 result_df,
#                 pd.DataFrame([total_row], index=['Total score']),
#                 pd.DataFrame([avg_row], index=['Average score'])
#             ])
        
#         return {
#             'results': result_df,
#             'data': self.data,
#             'selected_vars': selected_vars
#         }
    
#     def _table_stack_grouped(self, selected_df, selected_vars, by_var, iqr,
#                             decimal, test, name_test, total_column, sample_size,
#                             prevalence, percent, frequency, var_labels,
#                             vars_to_factor, assumption_p_value, simulate_p_value):
        
#         by_data = self.data[by_var].astype('category')
#         groups = by_data.cat.categories
        
#         results = []
        
#         # Add sample size row
#         if sample_size:
#             sample_row = {'Variable': 'Total'}
#             for grp in groups:
#                 sample_row[str(grp)] = (by_data == grp).sum()
#             if total_column:
#                 sample_row['Total'] = len(by_data)
#             if test:
#                 sample_row['P value'] = ''
#                 if name_test:
#                     sample_row['Test stat.'] = ''
#             results.append(sample_row)
#             results.append({k: '' for k in sample_row.keys()})
        
#         # Determine IQR variables automatically
#         iqr_vars = []
#         if iqr == "auto":
#             for var in selected_vars:
#                 if pd.api.types.is_numeric_dtype(selected_df[var]):
#                     # Test normality and homogeneity of variance
#                     groups_data = [selected_df[var][by_data == grp].dropna() for grp in groups]
#                     groups_data = [g for g in groups_data if len(g) >= 3]
                    
#                     if len(groups_data) >= 2:
#                         try:
#                             # Levene's test for homogeneity of variance
#                             _, p_levene = stats.levene(*groups_data)
                            
#                             # Shapiro-Wilk test on combined residuals
#                             combined = np.concatenate(groups_data)
#                             if len(combined) < 5000 and len(combined) >= 3:
#                                 _, p_shapiro = stats.shapiro(combined[:min(len(combined), 5000)])
                                
#                                 if p_shapiro < assumption_p_value or p_levene < assumption_p_value:
#                                     iqr_vars.append(var)
#                         except:
#                             pass
#         elif isinstance(iqr, list):
#             iqr_vars = iqr
        
#         # Process each variable
#         for var in selected_vars:
#             var_data = selected_df[var]
#             row = {'Variable': var}
            
#             # Get variable label
#             if var_labels and var in self.var_labels:
#                 row['Variable'] = self.var_labels[var]
            
#             # Categorical variable
#             if pd.api.types.is_categorical_dtype(var_data) or pd.api.types.is_object_dtype(var_data):
#                 crosstab = pd.crosstab(var_data, by_data)
                
#                 for grp in groups:
#                     if grp in crosstab.columns:
#                         counts = crosstab[grp]
#                         if percent == "column":
#                             pcts = (counts / counts.sum() * 100).round(decimal)
#                         else:
#                             pcts = (counts / crosstab.sum(axis=1) * 100).round(decimal)
                        
#                         if frequency:
#                             row[str(grp)] = f"{counts.values[0]} ({pcts.values[0]}%)"
#                         else:
#                             row[str(grp)] = f"{pcts.values[0]}%"
                
#                 if total_column:
#                     total_counts = crosstab.sum(axis=1)
#                     row['Total'] = str(total_counts.values[0])
                
#                 # Statistical test
#                 if test and len(groups) > 1:
#                     try:
#                         chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
                        
#                         # Check if Fisher's exact is more appropriate
#                         if (expected < 5).sum() / expected.size > 0.2:
#                             # For 2x2 tables
#                             if crosstab.shape == (2, 2):
#                                 _, p_value = stats.fisher_exact(crosstab)
#                                 test_name = "Fisher's exact test"
#                             else:
#                                 test_name = "Fisher's exact test"
#                         else:
#                             test_name = f"Chisq. ({dof} df) = {chi2:.{decimal+1}f}"
                        
#                         row['P value'] = f"< 0.001" if p_value < 0.001 else f"{p_value:.{decimal+2}f}"
#                         if name_test:
#                             row['Test stat.'] = test_name
#                     except:
#                         row['P value'] = 'NA'
#                         if name_test:
#                             row['Test stat.'] = 'Error'
            
#             # Numeric variable
#             elif pd.api.types.is_numeric_dtype(var_data):
#                 groups_data = []
                
#                 for grp in groups:
#                     grp_data = var_data[by_data == grp].dropna()
#                     groups_data.append(grp_data)
                    
#                     if var in iqr_vars:
#                         q1, q2, q3 = grp_data.quantile([0.25, 0.5, 0.75])
#                         row[str(grp)] = f"{q2:.{decimal}f} ({q1:.{decimal}f},{q3:.{decimal}f})"
#                     else:
#                         mean_val = grp_data.mean()
#                         sd_val = grp_data.std()
#                         row[str(grp)] = f"{mean_val:.{decimal}f} ({sd_val:.{decimal}f})"
                
#                 if total_column:
#                     if var in iqr_vars:
#                         q1, q2, q3 = var_data.quantile([0.25, 0.5, 0.75])
#                         row['Total'] = f"{q2:.{decimal}f} ({q1:.{decimal}f},{q3:.{decimal}f})"
#                     else:
#                         mean_val = var_data.mean()
#                         sd_val = var_data.std()
#                         row['Total'] = f"{mean_val:.{decimal}f} ({sd_val:.{decimal}f})"
                
#                 # Statistical test
#                 if test and len([g for g in groups_data if len(g) >= 3]) >= 2:
#                     try:
#                         if var in iqr_vars:
#                             # Non-parametric tests
#                             if len(groups) > 2:
#                                 stat, p_value = stats.kruskal(*groups_data)
#                                 test_name = "Kruskal-Wallis test"
#                             else:
#                                 stat, p_value = stats.mannwhitneyu(groups_data[0], groups_data[1])
#                                 test_name = "Ranksum test"
#                         else:
#                             # Parametric tests
#                             if len(groups) > 2:
#                                 stat, p_value = stats.f_oneway(*groups_data)
#                                 test_name = f"ANOVA F-test"
#                             else:
#                                 stat, p_value = stats.ttest_ind(groups_data[0], groups_data[1])
#                                 test_name = f"t-test ({len(groups_data[0])+len(groups_data[1])-2} df) = {abs(stat):.{decimal+1}f}"
                        
#                         row['P value'] = f"< 0.001" if p_value < 0.001 else f"{p_value:.{decimal+2}f}"
#                         if name_test:
#                             row['Test stat.'] = test_name
#                     except:
#                         row['P value'] = 'NA'
#                         if name_test:
#                             row['Test stat.'] = 'Sample too small'
            
#             results.append(row)
#             results.append({k: '' for k in row.keys()})  # Blank row
        
#         return pd.DataFrame(results)
    
# def tablestack(data: pd.DataFrame,
#                vars: List[Union[str, int]],
#                minlevel: Union[str, int] = "auto",
#                maxlevel: Union[str, int] = "auto",
#                count: bool = True,
#                na_rm: bool = False,
#                means: bool = True,
#                medians: bool = False,
#                sds: bool = True,
#                decimal: int = 1,
#                total: bool = True,
#                var_labels: bool = True,
#                var_labels_trunc: int = 150,
#                reverse: bool = False,
#                vars_to_reverse: Optional[List[Union[str, int]]] = None,
#                by: Optional[Union[str, int]] = None,
#                vars_to_factor: Optional[List[Union[str, int]]] = None,
#                iqr: Union[str, List[Union[str, int]]] = "auto",
#                prevalence: bool = False,
#                percent: str = "column",
#                frequency: bool = True,
#                test: bool = True,
#                name_test: bool = True,
#                total_column: bool = False,
#                simulate_p_value: bool = False,
#                sample_size: bool = True,
#                assumption_p_value: float = 0.01) -> Union[pd.DataFrame, Dict[str, Any]]:
    
#     ts = TableStack(data)
#     return ts.tablestack(
#         vars=vars, minlevel=minlevel, maxlevel=maxlevel, count=count,
#         na_rm=na_rm, means=means, medians=medians, sds=sds, decimal=decimal,
#         total=total, var_labels=var_labels, var_labels_trunc=var_labels_trunc,
#         reverse=reverse, vars_to_reverse=vars_to_reverse, by=by,
#         vars_to_factor=vars_to_factor, iqr=iqr, prevalence=prevalence,
#         percent=percent, frequency=frequency, test=test, name_test=name_test,
#         total_column=total_column, simulate_p_value=simulate_p_value,
#         sample_size=sample_size, assumption_p_value=assumption_p_value
#     )