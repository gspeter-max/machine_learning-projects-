import argparse
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math      
import numpy as np 
import sys       

parse_object = argparse.ArgumentParser( description = ' use thing ' )

parse_object.add_argument(
    'path',
    type = str,
    help = ' path of the file as a input ( eg ... content/data.csv ) '
)

parse_object.add_argument(
    '--describe',
    action = 'store_true', 
    help = 'if you like for describe tha data  ( eg.. --describe)'
)

parse_object.add_argument(
    '--plot',
    action = 'store_true', 
    help = 'if you like ploting run ( --plot) with run commend '
)

parse_object.add_argument(
    '--filter',
    action = 'store_true', 
    help = ' if you like to apply filtering run ( --filter ) with run commend  '
)

parse_object.add_argument(
    '--info',
    action = 'store_true', 
    help = ' if you like to roughly see the data run ( eg .. --info ) with run commend'
)

parse_object.add_argument(
    '--threshold',
    type = float,
    default = 2.0,
    help = 'threhold for applying in filter ( default : 2.0 ) '
)


try:
    access_object = parse_object.parse_args()
except argparse.ArgumentError as e:
     print(f"Error parsing arguments: {e}")
     parse_object.print_help()
     sys.exit(1)



path_ = access_object.path
decribe_ = access_object.describe 
ploting_ = access_object.plot
filter_ = access_object.filter
info_ = access_object.info
theshold_ = access_object.threshold 


try:

    print(f"Attempting to load data from: {path_}")
    df = pd.read_csv(path_)
    print(f"--- Successfully loaded '{path_}' ---")

except FileNotFoundError:
    print(f"Error: File not found at path '{path_}'. Check path and filename.") 
    sys.exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: DataFrame read from '{path_}' is empty. Check the file.") 
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred reading the file: {e}")
     sys.exit(1)


try:
    if describe_: 
        print("\n--- DataFrame Description ---")
        print(df.describe())
        print("-" * 20)

    
    if ploting_: 
        print("\n--- Plotting Distributions ---")
        try:

            numeric_cols = df.select_dtypes(include=np.number).columns
            N = len(numeric_cols)
            if N == 0 :
                print('Warning: No numeric columns found in DataFrame to plot.')
            else:

                rows = math.floor(np.sqrt(N))
                if rows == 0: rows = 1 
                cols = math.ceil(N / rows)

                print(f"Creating a {rows}x{cols} plot grid for {N} numeric columns...")
                fig, axes = plt.subplots(rows, cols ,figsize = (cols * 4, rows * 3)) 


                if N == 1:
                    axes_flat = [axes]
                else:
                    axes_flat = axes.flatten()

                col_iter = iter(numeric_cols) 

                plot_count = 0
                for i in range(rows):
                    for j in range(cols):

                        current_axis = axes_flat[plot_count]

                        try:
                           col_name = next(col_iter) 
                           if not df[col_name].isnull().all(): 

                                sns.histplot(df[col_name], ax=current_axis, kde=True)
                                current_axis.set_title(col_name)
                           else:
                               current_axis.set_title(f'{col_name} (All Null)')
                           plot_count += 1
                           if plot_count >= N: break 

                        except StopIteration:
                           current_axis.set_visible(False)
                           break
                        except Exception as plot_e:
                            print(f"Error plotting column (skipping): {plot_e}")
                            current_axis.set_title("Plot Error")

                    if plot_count >= N: break 


                for k in range(plot_count, len(axes_flat)):
                    axes_flat[k].set_visible(False)


                plt.tight_layout()
                plt.show()

        except Exception as e :
            print(f'\nAn error occurred during plotting: {e}')


    if filter_: 
        print(f"\n--- Filtering Numeric Columns (threshold = {theshold_}) ---")
        random.seed(42)

        
        def filtering(x):
            
            if isinstance(x, (int, float, np.number)):
                 return x > theshold_ 
            return False 


        try:
            original_row_count = len(df)
            print(f"Original row count: {original_row_count}")
            numeric_cols_to_filter = df.select_dtypes(include = 'number').columns
            if not numeric_cols_to_filter.empty:
                print(f"Applying filter to columns: {list(numeric_cols_to_filter)}")
                for col in numeric_cols_to_filter:
                    
                    if not df[col].isnull().all(): 


                        filtering_list = list(filter(filtering, df[col]))
                        v = filtering_list + (original_row_count - len(filtering_list)) *[None]
                        if len(v) == original_row_count:
                           df[col] = v
                        else:
                           print(f"Warning: Length mismatch after filtering column {col}. Skipping assignment.")
                    else:
                        print(f"Skipping column '{col}' as it is empty or all null.")

                print("\nDropping rows with any None/NaN values introduced by filtering...")
                df = df.dropna()
                print(f"Filtered dataframe final row count: {len(df)}")
            else:
                print("No numeric columns found to apply filter.")

        except Exception as e :
            print(f'\nAn error occurred during filtering: {e} ')
        print("-" * 20)


    if info_: 
        print("\n--- DataFrame Info ---")
        try:
            df.info()

        except Exception as e :
            print(f'\nAn error occurred getting DataFrame info: {e} ')
        print("-" * 20)

    print("\nScript finished processing.")


except Exception as e:
    print(f"\nAn unexpected error occurred during optional actions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
