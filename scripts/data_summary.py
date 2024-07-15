import pandas as pd
class DataFrameSummary:
    def __init__(self, csv_file_path):
        try:
            self.df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{csv_file_path}' was not found.")
        except Exception as e:
            raise Exception(f"Error reading CSV file '{csv_file_path}': {str(e)}")
    
    def summarize_data(self):
        try:
            numeric_stats = self.df.describe().transpose()
            missing_values = self.df.isnull().sum()
            summary_df = pd.DataFrame(index=self.df.columns)
            summary_df['Data Type'] = self.df.dtypes
            summary_df['Missing Values'] = missing_values
            summary_df['Unique Values'] = self.df.nunique()
            summary_df = summary_df.join(numeric_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
            return summary_df
        except Exception as e:
            raise Exception(f"Error summarizing data: {str(e)}")

    def perform_value_counts(self):
        try:
            value_counts_dict = {}
            for column_name in self.df.columns:
                value_counts = self.df[column_name].value_counts().reset_index()
                value_counts.columns = ['Value', 'Count']
                value_counts_dict[column_name] = value_counts.set_index('Value')['Count']
            
            return pd.DataFrame(value_counts_dict)
        except Exception as e:
            raise Exception(f"Error performing value counts: {str(e)}")
