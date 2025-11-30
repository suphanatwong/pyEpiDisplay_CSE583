import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, List, Optional, Dict, Any
import warnings


class TableStack:
    
    def __init__(self, data: pd.DataFrame):

        self.data = data.copy()
        self.var_labels = {}
        
    def set_var_labels(self, labels: Dict[str, str]):
        self.var_labels = labels
        
    def tableStack(self,
                   vars: List[Union[str, int]],
                   minlevel: Union[str, int] = "auto",
                   maxlevel: Union[str, int] = "auto",
                   count: bool = True,
                   na_rm: bool = False,
                   means: bool = True,
                   medians: bool = False,
                   sds: bool = True,
                   decimal: int = 1,
                   total: bool = True,
                   var_labels: bool = True,
                   var_labels_trunc: int = 150,
                   reverse: bool = False,
                   vars_to_reverse: Optional[List[Union[str, int]]] = None,
                   by: Optional[Union[str, int]] = None,
                   vars_to_factor: Optional[List[Union[str, int]]] = None,
                   iqr: Union[str, List[Union[str, int]]] = "auto",
                   prevalence: bool = False,
                   percent: str = "column",
                   frequency: bool = True,
                   test: bool = True,
                   name_test: bool = True,
                   total_column: bool = False,
                   simulate_p_value: bool = False,
                   sample_size: bool = True,
                   assumption_p_value: float = 0.01) -> Union[pd.DataFrame, Dict[str, Any]]:

        # Convert variable names to list if needed
        if isinstance(vars, (str, int)):
            vars = [vars]
            
        # Convert column indices (column poisition) to names if needed
        selected_vars = []
        for v in vars:
            if isinstance(v, int):
                selected_vars.append(self.data.columns[v])
            else:
                selected_vars.append(v)
                
        # Handle vars_to_factor
        if vars_to_factor is not None:
            vars_to_factor_names = []
            for v in vars_to_factor:
                if isinstance(v, int): #check if index
                    vars_to_factor_names.append(self.data.columns[v])
                else:
                    vars_to_factor_names.append(v)
                    
            for var in vars_to_factor_names:
                if var in selected_vars:
                    self.data[var] = self.data[var].astype('category')
        
        # Extract selected data
        selected_df = self.data[selected_vars].copy()
        
        # Case 1: No grouping variable
        if by is None:
            return self._table_stack_ungrouped(
                selected_df, selected_vars, minlevel, maxlevel, count,
                na_rm, means, medians, sds, decimal, total, var_labels,
                var_labels_trunc, reverse, vars_to_reverse
            )
        
        # Case 2: With grouping variable
        else:
            if isinstance(by, int):
                by_var = self.data.columns[by]
            else:
                by_var = by
                
            return self._table_stack_grouped(
                selected_df, selected_vars, by_var, iqr, decimal,
                test, name_test, total_column, sample_size, prevalence,
                percent, frequency, var_labels, vars_to_factor,
                assumption_p_value, simulate_p_value
            )
    
    def _table_stack_ungrouped(self, selected_df, selected_vars, minlevel,
                               maxlevel, count, na_rm, means, medians, sds,
                               decimal, total, var_labels, var_labels_trunc,
                               reverse, vars_to_reverse):
        
        results = []
        
        # Determine min/max levels
        if minlevel == "auto":
            minlevel = selected_df.select_dtypes(include=[np.number]).min().min()
        if maxlevel == "auto":
            maxlevel = selected_df.select_dtypes(include=[np.number]).max().max()
            
        for var in selected_vars:
            row_data = {}
            col_data = selected_df[var]
            
            # Handle categorical/factor variables
            if pd.api.types.is_categorical_dtype(col_data) or pd.api.types.is_object_dtype(col_data):  #isinstance is too strict
                value_counts = col_data.value_counts()
                for level, cnt in value_counts.items():
                    row_data[str(level)] = cnt
            else:
                # Numeric variable - create levels
                if minlevel != "auto" and maxlevel != "auto":
                    for level in range(int(minlevel), int(maxlevel) + 1):
                        row_data[str(level)] = (col_data == level).sum()
            
            # Add statistics
            if count:
                row_data['count'] = col_data.notna().sum()
                
            if pd.api.types.is_numeric_dtype(col_data):
                if means:
                    row_data['mean'] = round(col_data.mean(), decimal)
                if medians:
                    row_data['median'] = round(col_data.median(), decimal)
                if sds:
                    row_data['sd'] = round(col_data.std(), decimal)
            
            # Add variable label
            if var_labels and var in self.var_labels:
                row_data['description'] = self.var_labels[var][:var_labels_trunc]
                
            results.append(row_data)
        
        # Create DataFrame
        result_df = pd.DataFrame(results, index=selected_vars)
        
        # Add total scores if numeric
        if total and all(pd.api.types.is_numeric_dtype(selected_df[v]) for v in selected_vars):
            total_scores = selected_df.sum(axis=1)
            avg_scores = selected_df.mean(axis=1)
            
            total_row = {
                'count': total_scores.notna().sum(),
                'mean': round(total_scores.mean(), decimal),
                'sd': round(total_scores.std(), decimal)
            }
            avg_row = {
                'count': avg_scores.notna().sum(),
                'mean': round(avg_scores.mean(), decimal),
                'sd': round(avg_scores.std(), decimal)
            }
            
            result_df = pd.concat([
                result_df,
                pd.DataFrame([total_row], index=['Total score']),
                pd.DataFrame([avg_row], index=['Average score'])
            ])
        
        return {
            'results': result_df,
            'data': self.data,
            'selected_vars': selected_vars
        }
    
    def _table_stack_grouped(self, selected_df, selected_vars, by_var, iqr,
                            decimal, test, name_test, total_column, sample_size,
                            prevalence, percent, frequency, var_labels,
                            vars_to_factor, assumption_p_value, simulate_p_value):
        
        by_data = self.data[by_var].astype('category')
        groups = by_data.cat.categories
        
        results = []
        
        # Add sample size row
        if sample_size:
            sample_row = {'Variable': 'Total'}
            for grp in groups:
                sample_row[str(grp)] = (by_data == grp).sum()
            if total_column:
                sample_row['Total'] = len(by_data)
            if test:
                sample_row['P value'] = ''
                if name_test:
                    sample_row['Test stat.'] = ''
            results.append(sample_row)
            results.append({k: '' for k in sample_row.keys()})
        
        # Determine IQR variables automatically
        iqr_vars = []
        if iqr == "auto":
            for var in selected_vars:
                if pd.api.types.is_numeric_dtype(selected_df[var]):
                    # Test normality and homogeneity of variance
                    groups_data = [selected_df[var][by_data == grp].dropna() for grp in groups]
                    groups_data = [g for g in groups_data if len(g) >= 3]
                    
                    if len(groups_data) >= 2:
                        try:
                            # Levene's test for homogeneity of variance
                            _, p_levene = stats.levene(*groups_data)
                            
                            # Shapiro-Wilk test on combined residuals
                            combined = np.concatenate(groups_data)
                            if len(combined) < 5000 and len(combined) >= 3:
                                _, p_shapiro = stats.shapiro(combined[:min(len(combined), 5000)])
                                
                                if p_shapiro < assumption_p_value or p_levene < assumption_p_value:
                                    iqr_vars.append(var)
                        except:
                            pass
        elif isinstance(iqr, list):
            iqr_vars = iqr
        
        # Process each variable
        for var in selected_vars:
            var_data = selected_df[var]
            row = {'Variable': var}
            
            # Get variable label
            if var_labels and var in self.var_labels:
                row['Variable'] = self.var_labels[var]
            
            # Categorical variable
            if pd.api.types.is_categorical_dtype(var_data) or pd.api.types.is_object_dtype(var_data):
                crosstab = pd.crosstab(var_data, by_data)
                
                for grp in groups:
                    if grp in crosstab.columns:
                        counts = crosstab[grp]
                        if percent == "column":
                            pcts = (counts / counts.sum() * 100).round(decimal)
                        else:
                            pcts = (counts / crosstab.sum(axis=1) * 100).round(decimal)
                        
                        if frequency:
                            row[str(grp)] = f"{counts.values[0]} ({pcts.values[0]}%)"
                        else:
                            row[str(grp)] = f"{pcts.values[0]}%"
                
                if total_column:
                    total_counts = crosstab.sum(axis=1)
                    row['Total'] = str(total_counts.values[0])
                
                # Statistical test
                if test and len(groups) > 1:
                    try:
                        chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
                        
                        # Check if Fisher's exact is more appropriate
                        if (expected < 5).sum() / expected.size > 0.2:
                            # For 2x2 tables
                            if crosstab.shape == (2, 2):
                                _, p_value = stats.fisher_exact(crosstab)
                                test_name = "Fisher's exact test"
                            else:
                                test_name = "Fisher's exact test"
                        else:
                            test_name = f"Chisq. ({dof} df) = {chi2:.{decimal+1}f}"
                        
                        row['P value'] = f"< 0.001" if p_value < 0.001 else f"{p_value:.{decimal+2}f}"
                        if name_test:
                            row['Test stat.'] = test_name
                    except:
                        row['P value'] = 'NA'
                        if name_test:
                            row['Test stat.'] = 'Error'
            
            # Numeric variable
            elif pd.api.types.is_numeric_dtype(var_data):
                groups_data = []
                
                for grp in groups:
                    grp_data = var_data[by_data == grp].dropna()
                    groups_data.append(grp_data)
                    
                    if var in iqr_vars:
                        q1, q2, q3 = grp_data.quantile([0.25, 0.5, 0.75])
                        row[str(grp)] = f"{q2:.{decimal}f} ({q1:.{decimal}f},{q3:.{decimal}f})"
                    else:
                        mean_val = grp_data.mean()
                        sd_val = grp_data.std()
                        row[str(grp)] = f"{mean_val:.{decimal}f} ({sd_val:.{decimal}f})"
                
                if total_column:
                    if var in iqr_vars:
                        q1, q2, q3 = var_data.quantile([0.25, 0.5, 0.75])
                        row['Total'] = f"{q2:.{decimal}f} ({q1:.{decimal}f},{q3:.{decimal}f})"
                    else:
                        mean_val = var_data.mean()
                        sd_val = var_data.std()
                        row['Total'] = f"{mean_val:.{decimal}f} ({sd_val:.{decimal}f})"
                
                # Statistical test
                if test and len([g for g in groups_data if len(g) >= 3]) >= 2:
                    try:
                        if var in iqr_vars:
                            # Non-parametric tests
                            if len(groups) > 2:
                                stat, p_value = stats.kruskal(*groups_data)
                                test_name = "Kruskal-Wallis test"
                            else:
                                stat, p_value = stats.mannwhitneyu(groups_data[0], groups_data[1])
                                test_name = "Ranksum test"
                        else:
                            # Parametric tests
                            if len(groups) > 2:
                                stat, p_value = stats.f_oneway(*groups_data)
                                test_name = f"ANOVA F-test"
                            else:
                                stat, p_value = stats.ttest_ind(groups_data[0], groups_data[1])
                                test_name = f"t-test ({len(groups_data[0])+len(groups_data[1])-2} df) = {abs(stat):.{decimal+1}f}"
                        
                        row['P value'] = f"< 0.001" if p_value < 0.001 else f"{p_value:.{decimal+2}f}"
                        if name_test:
                            row['Test stat.'] = test_name
                    except:
                        row['P value'] = 'NA'
                        if name_test:
                            row['Test stat.'] = 'Sample too small'
            
            results.append(row)
            results.append({k: '' for k in row.keys()})  # Blank row
        
        return pd.DataFrame(results)
    
def tableStack(data: pd.DataFrame,
               vars: List[Union[str, int]],
               minlevel: Union[str, int] = "auto",
               maxlevel: Union[str, int] = "auto",
               count: bool = True,
               na_rm: bool = False,
               means: bool = True,
               medians: bool = False,
               sds: bool = True,
               decimal: int = 1,
               total: bool = True,
               var_labels: bool = True,
               var_labels_trunc: int = 150,
               reverse: bool = False,
               vars_to_reverse: Optional[List[Union[str, int]]] = None,
               by: Optional[Union[str, int]] = None,
               vars_to_factor: Optional[List[Union[str, int]]] = None,
               iqr: Union[str, List[Union[str, int]]] = "auto",
               prevalence: bool = False,
               percent: str = "column",
               frequency: bool = True,
               test: bool = True,
               name_test: bool = True,
               total_column: bool = False,
               simulate_p_value: bool = False,
               sample_size: bool = True,
               assumption_p_value: float = 0.01) -> Union[pd.DataFrame, Dict[str, Any]]:
    
    ts = TableStack(data)
    return ts.tableStack(
        vars=vars, minlevel=minlevel, maxlevel=maxlevel, count=count,
        na_rm=na_rm, means=means, medians=medians, sds=sds, decimal=decimal,
        total=total, var_labels=var_labels, var_labels_trunc=var_labels_trunc,
        reverse=reverse, vars_to_reverse=vars_to_reverse, by=by,
        vars_to_factor=vars_to_factor, iqr=iqr, prevalence=prevalence,
        percent=percent, frequency=frequency, test=test, name_test=name_test,
        total_column=total_column, simulate_p_value=simulate_p_value,
        sample_size=sample_size, assumption_p_value=assumption_p_value
    )