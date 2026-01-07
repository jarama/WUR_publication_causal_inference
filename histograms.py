import pandas as pd
import functions as cfun

#region <INITIALIZE TERMINAL OUTPUT>

# Just some print output to keep the reminal ordered.

print("\n")
print("=== histograms.py info ===")
print("─" * 75)

#endregion

#region <SIMULATION NAME>

name = "test1"                                   # Set name of the simulation run

#endregion

#region <PATHS>

# Set the paths to the correct directories and files that contain the  
# simulation and summary data. These should be saved in data/processed.

simulated = "data/processed/"+name+"_sim.csv.gz"         # Set simulation data path
summary = "data/processed/"+name+"_sum.csv.gz"              # Set summary data path

#endregion

#region <LOAD DATA>

# Read the data as Pandas dataframe

df_summary = pd.read_csv(summary, compression="gzip")
df_simulated = pd.read_csv(simulated, compression="gzip")

#endregion

#region <SAVE TABLE>  

# The function protected_save(is_table=True) saves the summary data as a HTML    
# file. This results in a table that is a easier to read than a terminal print.
# It can be opend in your browser.
  
cfun.protected_save(df_summary, 
                    filename=name+".html",             # Set filename 
                    out_dir="summary",      # Set output directory 
                    is_table=True)

#endregion

#region <PLOT HISTOGRAMS>

# Set the parameter sets that you want to visualize as histograms. Don't worry
# if some combinations do not exist in your data. These are automatically skipped.
# It is not necessary to set the sample size (n) or the standard deviation of Y (sd_Y).
# These are automatically split in the plot function.  
# MAKE SURE TO NOT CHANGE THE FORMAT!!!

param_sets = {                                              # Set parameters to plot
    "beta_0": [0, 1/3, 1, 3],
    "beta_1": [0],
    "beta_2": [0],
    "beta_3": [0],
    "beta_4": [0],
    "beta_5": [0],
    "beta_6": [0],
    "beta_7": [0],
    "beta_8": [0],
    "incl_var": ["('T',)",
                 "('T', 'M_1', 'M_2', 'Z_1', 'Z_2', 'C')"]  
}

#endregion

#region <PLOT HISTOGRAMS>

# The function plot_param_sets() makes histogram plots of the input data and 
# parameters. The plots are saved to the set output directory. If desired,
# confidence intervals (based on the simulation outcomes; thus empirical) can be 
# turned on (default) or off. These plots are quite rough, and are not the final 
# fine-tuned figures. However, they do provide the necessary insights into the 
# data and are sufficient for discussions.    

cfun.plot_param_sets(df_simulated, param_sets, 
                     confidence=True,                       # Confidence intervals on (True; default) or off (False)
                     out_dir="figures/"+name      # Set output directory
                     )     

#endregion
