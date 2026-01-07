import numpy as np
import pandas as pd
import functions as cfun

# This script is used to simulate and estimate over a grid of settings.
# Read the instructions in the script BEFORE using AND the instructions/output
# in the terminal carfully! This should guide you throught the process.

if __name__ == "__main__":

    #region <INITIALIZE TERMINAL OUTPUT>

    # Just some print output to keep the reminal ordered.
 
    print("\n")
    print("=== system_simulations.py info ===")
    print("─" * 75)

    #endregion

    #region <SYSTEM VIZUALIZATION>

    # The function visualize_system() makes a Graphviz diagram.
    # If run=False, the function is skipped. It is recommended to 
    # only run this function once, to prevent it from useless processing.
   
    cfun.visualize_system(run=False)
    
    #endregion

    #region <SIMULATION NAME>

    name = "test1"                                   # Set name of the simulation run
    
    #endregion

    #region <SIMULATION GRID>

    # Here, the simulation and estimation grid can be set. Just follow the 
    # current format. Comment and uncomment wherever necessary and set your
    # own desired grid. MAKE SURE TO NOT CHANGE THE FORMAT!!! 

    # Grids
    grid_params = {                                 # Set beta coefficients
        "beta_0": [0, 1/3, 1, 3],   
        "beta_1": [0],
        "beta_2": [0],
        "beta_3": [0],
        "beta_4": [0],
        "beta_5": [0],
        "beta_6": [0],
        "beta_7": [0],
        "beta_8": [0],
        "sd_Y": [1/3, 1, 3]
    }

    variable_sets = [                               # Set variable sets to control (AKA regressors)
        ["T"],
        # ["T", "M_1", "M_2"], 
        # ["T", "M_1", "M_2", "Z_1", "Z_2", "C"]  
    ]

    n_list = [10, 100, 1000]                        # Set sample size(s) 
    
    n_iter = 1000                                   # Set the amount of iterations

    excluded_param_sets = [                         # Set coefficients to exclude
        # {"beta_2": 0, "beta_3": 0},
        # {"beta_2": 0, "beta_4": 1/3}, 
        # {"beta_2": 0, "beta_4": 1}, 
        # {"beta_2": 0, "beta_4": 3},                   
    ]

    excluded_var_sets = [                           # Set specific variable sets to exclude
        # ("T", "M_1")
    ]
    
    #endregion

    #region <SIMULATIONS>

    # Set a seed in the Random Number Generator (I use 1001).
    # Also provide the number of cores you want to use for the simulations
    # and estimations. If you forget this, half of the available cores
    # will be used. Don't change any other settings, these are all defined 
    # before in this script.  

    rng = np.random.default_rng(seed=1001)          # Set seed

    sim_df = cfun.parallel_simulation(
        grid_params=grid_params,
        variable_sets=variable_sets,
        n_list=n_list,
        iterations=n_iter,
        rng=rng,
        excluded_param_sets=excluded_param_sets,
        excluded_var_sets=excluded_var_sets,
        n_jobs=12                                    # Set cores
    )
    
    #endregion

    #region <SUMMARIZE>

    # Just leave this. It will summarize the simulation results for you.

    summary_df = cfun.summarize(sim_df)
    
    #endregion

    #region <SAVE>

    # This function will save the simulation results and summary to the 
    # data/processed directory. Provide a logical name for the current run so 
    # you remember what settings you used. It is also recommended to save
    # the grid that you used somewhere so you can always refer back to them.
    # GRID AND OTHER SETTINGS ARE NOT AUTOMATICALLY SAVED!

    cfun.save_sim(sim_df, 
                  summary_df,
                  sim_name=name+"_sim",       # Set name of simulation results
                  summary_name=name+"_sum"       # Set name of summary
                )
    
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

    # The function plot_param_sets() makes histogram plots of the input data and 
    # parameters. The plots are saved to the set output directory. If desired,
    # confidence intervals (based on the simulation outcomes; thus empirical) can be 
    # turned on (default) or off. These plots are quite rough, and are not the final 
    # fine-tuned figures. However, they do provide the necessary insights into the 
    # data and are sufficient for discussions.    


    cfun.plot_param_sets(df_simulated,
                        confidence=False,                       # Confidence intervals on (True) or off (False; default)
                        out_dir="figures/"+name      # Set output directory
                        )     

    #endregion
