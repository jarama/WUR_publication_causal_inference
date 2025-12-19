
import numpy as np
import pandas as pd
import graphviz as gr
import os
import shutil
from multiprocessing import Pool, cpu_count
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

#region <def FIGURE>
def visualize_system(filename="system_diagram", run=True):
    """
    Creates a diagram of the system using Graphviz.
    """
    # Title in terminal
    print("--- System visualization ---")
    
    # Function ON/OFF parameter
    if run is False:
       print("Function visualize_system() was not executed...")
       print("─" * 75)
       return None

    # Verify Graphviz installation 
    if shutil.which("dot") is None:
        print("⚠️ WARNING: Graphviz is not installed on device or not on PATH. Diagram generation was not executed!")
        print("─" * 75)
        return None

    # Create output folder if needed
    out_dir = os.path.join("figures", "system_diagrams")
    os.makedirs(out_dir, exist_ok=True)

    # filename cleanup (remove extensions if given)
    base = os.path.splitext(filename)[0]

    # Final save paths
    png_path = os.path.join(out_dir, base + ".png")
    svg_path = os.path.join(out_dir, base + ".svg")

    # System diagram
    g = gr.Digraph(filename=base)
    g.attr(rankdir='LR')

    g.edge("C", "T", label="β₅")
    g.edge("Z₂", "M₁", label="β₄")
    g.edge("T", "M₁", label="β₂")
    g.edge("T", "M₂", label="β₃")

    g.edge("Z₁", "Y", label="β₁")
    g.edge("M₁", "Y", label="β₇")
    g.edge("M₂", "Y", label="β₈")
    g.edge("C", "Y", label="β₆")
    g.edge("T", "Y", label="β₀")

    # Render PNG
    g.format = "png"
    g.render(filename=base, directory=out_dir, cleanup=True)

    # Render SVG
    g.format = "svg"
    g.render(filename=base, directory=out_dir, cleanup=True)

    # Output absolute paths
    png_abs = os.path.abspath(png_path)
    svg_abs = os.path.abspath(svg_path)

    print("System diagram saved to folder:")
    print(f"    {out_dir}")
    print("Saved files:")
    print(f"    PNG: {png_abs}")
    print(f"    SVG: {svg_abs}")
    print("─" * 75)

    return png_abs, svg_abs

#endregion

#region <def DATA GENERATION>
def generate_data(n,
                  rng, 
                  beta_0, 
                  beta_1, 
                  beta_2, 
                  beta_3, 
                  beta_4, 
                  beta_5, 
                  beta_6, 
                  beta_7, 
                  beta_8, 
                  sd_Y):
    """
    Generate data for simulations:

        Coefficients:
        beta_0 (T -> Y)
        beta_1 (Z_1 -> Y) 
        beta_2 (T -> M_1)
        beta_3 (T -> M_2)
        beta_4 (Z_2 -> M_1) 
        beta_5 (C -> T) 
        beta_6 (C -> Y) 
        beta_7 (M_1 -> Y) 
        beta_8 (M_2 -> Y)

        Other:
        n      =   sample size
        sd_Y   =   standard deviation of the noise/error term of variable Y  
    """
    # Exogenous variable generation
    C = rng.normal(0, 1, n)
    Z_1 = rng.normal(0, 1, n)
    Z_2 = rng.normal(0, 1, n)

    # Endogenous variable generation
    T = beta_5*C + rng.normal(0, 1, n)

    M_1 = beta_2*T + beta_4*Z_2 + rng.normal(0, 1, n)
    M_2 = beta_3*T + rng.normal(0, 1, n)

    Y = beta_0*T + beta_6*C + beta_7*M_1 + beta_8*M_2 + beta_1*Z_1 + rng.normal(0, sd_Y, n)

    # Causal effect calculation  
    dir_eff = beta_0
    tot_eff = beta_0 + (beta_2*beta_7) + (beta_3*beta_8)
    indir_eff = tot_eff - dir_eff

    # Y standard deviation value
    Y_sd = sd_Y

    return pd.DataFrame({# Causal effects
                        "total": tot_eff,
                        "direct": dir_eff,
                        "indirect": indir_eff,

                        # Y sd
                        "Y_sd": Y_sd,

                        # Expgenous
                        "C": C,
                        "Z_1": Z_1, 
                        "Z_2": Z_2,

                        # Endogenous
                        "T": T,
                        "M_1": M_1,
                        "M_2": M_2,
                        "Y": Y})

#endregion

#region <def SIMULATION>

# def Params report
def report_simulation_plan(grid_params, variable_sets, n_list, iterations,
                           excluded_param_sets=None, excluded_var_sets=None):
    """
    Reports the amount of combinations, excluded iterations, 
    and the amount of iterations that will actually be run.
    """
    # Set default exclutions
    if excluded_param_sets is None:
        excluded_param_sets = []
    if excluded_var_sets is None:
        excluded_var_sets = []

    # Get parameter names and combinations
    param_names = list(grid_params.keys())
    grid = list(product(*grid_params.values()))

    # Initialize parameter
    excluded_param_count = 0
    included_param_combos = []

    # Count included and excluded parameters
    for params in grid:
        param_dict = dict(zip(param_names, params))
        if any(all(param_dict.get(k) == v for k, v in excl.items()) for excl in excluded_param_sets):
            excluded_param_count += 1
        else:
            included_param_combos.append(param_dict)

    # Initialize variable sets
    excluded_varset_count = 0
    included_varsets = []

    # Count included and excluded variable sets
    for vs in variable_sets:
        if tuple(vs) in excluded_var_sets:
            excluded_varset_count += 1
        else:
            included_varsets.append(vs)

    # Total counts
    full_param = len(grid)
    full_varsets = len(variable_sets)

    # Amount of sets that will be included
    valid_param = full_param - excluded_param_count
    valid_varsets = full_varsets - excluded_varset_count

    # Total amount of runs
    total_full_runs = iterations * full_param * full_varsets * len(n_list)
    
    # Total amount of excluded runs
    total_skipped_runs = (
        iterations * excluded_param_count * full_varsets * len(n_list) +
        iterations * valid_param * excluded_varset_count * len(n_list)
    )

    # Total amount of included runs (actual runs)
    total_actual_runs = total_full_runs - total_skipped_runs

    # Message
    print("📊 SIMULATION RUN PLAN")
    print()
    print(f"Parameter combinations total:         {full_param}")
    print(f"Parameter combinations excluded:      {excluded_param_count}")
    print(f"→ valid parameter sets:               {valid_param}")
    print()
    print(f"Variable sets total:                  {full_varsets}")
    print(f"Variable sets excluded:               {excluded_varset_count}")
    print(f"→ valid variable sets:                {valid_varsets}")
    print()
    print(f"Total different sample sizes:         {len(n_list)}")
    print(f"Total iterations:                     {iterations}")
    print()
    print(f"Full possible runs (no exclusions):   {total_full_runs:,}")
    print(f"Runs avoided due to exclusions:       {total_skipped_runs:,}")
    print(f"💡 Total actual runs to be executed:  {total_actual_runs:,}")
    print()

    return total_actual_runs

# def Excldue
def excluded(param_dict, varset, excluded_param_sets, excluded_var_sets):
    """
    Returns True if passed parameter or variable set is in the excluded definitions.
    Otherwise, returns False.
    """

    # Compare parameters to exclutions
    for excl in excluded_param_sets:
        if all(param_dict.get(k) == v for k, v in excl.items()):
            return True

    # Compare variable sets to exclutions
    if varset is not None and tuple(varset) in excluded_var_sets:
        return True

    return False

# def Worker
def run_single_sim(args):
    """
    Defines a worker to run on a single core in parallel setup.
    """

    # Get arguments
    iter_id, params, n, param_names, var_set, rng_seed = args

    # Random number generator (seed)
    rng = np.random.default_rng(rng_seed)

    # Combine parameter names and values in dictionary
    param_dict = dict(zip(param_names, params))

    # Generate data based on sample size (n), seed and parameters 
    df = generate_data(n, rng, **param_dict)
    
    # Define outcome variable y
    y = df["Y"].to_numpy()

    # Define X (intercept, variables)
    X = np.column_stack([np.ones(n), df[list(var_set)].to_numpy()])

    # Matrix multiplocation for OLS
    XtX = X.T @ X
    Xty = X.T @ y

    # Solve a linear matrix equation
    beta = np.linalg.solve(XtX, Xty)

    # Get index of T and assign to coef_T  
    col_idx = ["intercept"] + list(var_set)
    coef_T = beta[col_idx.index("T")]

    # Return tuple with unpacked parameters and single values for direct, indirect, total and T_sd
    return [(iter_id, n, *params, var_set,
             coef_T,
             df["direct"].iat[0], df["indirect"].iat[0],
             df["total"].iat[0], df["Y_sd"].iat[0])]

# def Simulate and estimate
def parallel_simulation(
    grid_params, variable_sets, 
    n_list, iterations=10, 
    rng=None, excluded_param_sets=None,
    excluded_var_sets=None, n_jobs=None):
    """
    Runs parallel simulations and OLS estimations based on the settings defined in the input.
    """
    
    # Set default exclutions
    if excluded_param_sets is None:
        excluded_param_sets = []
    if excluded_var_sets is None:
        excluded_var_sets = []

    # Warn and exit if seed is not passed
    if rng is None:
        print("No random number generator was passed!")
        print("Simulations aborted...")
        exit()

    # If n_jobs is not set: be save and use half of the machines cores
    if n_jobs is None:
        n_jobs = max(1, cpu_count() // 2)

    # Warn and exit if set number of cores exeeds available cores
    if n_jobs > cpu_count():
        print(f"⚠️ You selected {n_jobs} cores! Only {cpu_count()} available!")
        print("Simulations aborted...")
        print("─" * 75)
        exit()

    # Warn and ask if number of set cores is equal to number of available cores  
    if n_jobs == cpu_count():
        answer_1 = input(f"⚠️ You selected all available cores ({n_jobs}/{cpu_count()})! Are you sure you want to proceed? [Y/n]: ").strip().lower()
        print() 
        if answer_1 == "n": 
            print("Simulations aborted...") 
            print("─" * 75) 
            exit() 
        else: 
            print("Starting simulations...", "\n")

    # Report on runs 
    total_actual_runs = report_simulation_plan(
        grid_params, variable_sets, n_list, iterations,
        excluded_param_sets, excluded_var_sets
    )

    # Ask approval to run, if not: exit
    answer_2 = input(f"You are about to run {total_actual_runs:,} simulations. Proceed? [Y/n]: ").strip().lower()
    print() 
    if answer_2 == "n": 
        print("Simulations aborted...") 
        print("─" * 75) 
        exit() 
    else: 
        print("Starting simulations...", "\n")
    
    # Get parameter names and combinations
    param_names = list(grid_params.keys())
    grid = list(product(*grid_params.values()))

    # Initialize job list
    jobs = []

    # Initialize seed sequence based on random number generator
    seed_seq = np.random.SeedSequence(rng.integers(1e9))

    # build job list (one job per iter-param-n combo)
    for (iter_id, params, n, var_set), seed in zip(
        product(range(iterations), grid, n_list, variable_sets),
        seed_seq.spawn(iterations * len(grid) * len(n_list) * len(variable_sets))):

        # Combine parameter names and values in dictionary
        param_dict = dict(zip(param_names, params))

        # If arguments are in exceptions: do not add to job list
        if excluded(param_dict, var_set, excluded_param_sets, excluded_var_sets):
            continue

        # Add grid variables to job list
        jobs.append((iter_id, params, n, param_names, var_set, seed.generate_state(1)[0]))

    # Message
    print(f"🧠 Parallelizing across {n_jobs} core(s)...")
    print(f"🚀 Jobs to run: {len(jobs):,}\n")

    # Dispatch to workers
    with Pool(n_jobs) as pool:
        results = list(tqdm(pool.imap(run_single_sim, jobs), total=len(jobs)))

    # Flatten results
    sim_data = [row for batch in results for row in batch]

    # Set feature names dynamically based on amound of regressors
    colnames = (
        ["iteration", "n"] + param_names +
        ["incl_vars", "coef_T", "direct_effect", "indirect_effect", "total_effect", "Y_sd"]
    )

    # Make pandas df
    sim_df = pd.DataFrame(sim_data, columns=colnames)

    # Finish message
    print("\n✔ Parallel simulation complete.\n")

    return sim_df

# def Summarize
def summarize(df):
    """
    Summarizes the results opf the estimations.
    """

    print("--- Summarize estimations ---")

    # Prepare incl_vars
    if "incl_vars" in df.columns:
            df["incl_vars"] = df["incl_vars"].apply(tuple)

    # Get list of coefficient names
    param_names = list(df.loc[:, "beta_0":"beta_8"].columns)

    # Summarize by grouping and aggregating 
    summary_df = (
        df
        .groupby(param_names + ["Y_sd", "incl_vars", "n"], as_index=False)
        .agg(
            T_est=("coef_T", "mean"),
            T_sd=("coef_T", lambda x: x.std(ddof=0)),
            direct_effect=("direct_effect", "first"),   
            indirect_effect=("indirect_effect", "first"), 
            total_effect=("total_effect", "first")
        )
    )
   
    # Message 
    n_rows = summary_df.shape[0]
    print(f"Estimation summary contains {n_rows} unique combinations.")
    print("─" * 75)

    return summary_df

#endregion

#region <def SAVE>

# def Extension
def ensure_extension(name: str) -> str:
        """
        Handles file extentions
        """
        
        # Remove if ".csv", ".gz"
        name = name.replace(".csv", "").replace(".gz", "").replace(".csv.gz", "")

        return f"{name}.csv.gz"

# def Protected save
def protected_save(df, filename, 
                   out_dir=os.getcwd(), 
                   is_table=False):
    """
    Save dataframe with overwrite protection and dynamic paths.

    Parameters:
        df (pd.DataFrame)
        filename (str) -> "file.csv" or "file"
        out_dir (str) -> save directory (default = current working directory)
        is_table (bool) -> if True, save as HTML table instead of CSV
    """

    # Handle extension based on table mode
    if is_table:
        if not filename.endswith(".html"):
            filename = filename + ".html"
    else:
        filename = ensure_extension(filename)

    # Normalize and create directory if not exist
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Full resolved path
    file_path = os.path.join(out_dir, filename)

    # Overwrite protection
    if os.path.exists(file_path):
        print(f"⚠️ File '{file_path}' already exists.")

        # Give user a choice to overwrite existing files (safety feature)
        choice = input("Do you want to overwrite it? [Y/n]: ").strip().lower()
        print()

        # If  choice is 'yes', overwrite files
        if choice == "y":
            if is_table:
                df.to_html(file_path, index=False)
            else:
                df.to_csv(file_path, index=False, compression="gzip")

            # Message
            print(f"✔ Overwritten: {file_path}\n")

            return file_path

        # If not overwrite: rename file, but save in same folder
        new_name = input("Enter a new filename (no extension needed): ").strip()
        
        return protected_save(df, new_name, out_dir, is_table)

    # If file does not exist → save normally
    if is_table:
        df.to_html(file_path, index=False)
    else:
        df.to_csv(file_path, index=False, compression="gzip")

    # Message
    print(f"✔ Saved: {file_path}\n")
    
    return file_path

# def Save
def save_sim(df_simulation, df_summary, 
             sim_name="simulation", 
             summary_name="summary", 
             out_dir="data/processed"):
    """
    Save simulation and summary data to input directory.

    Default directory: .../data/processed
    """
    
    print("--- Save dataframes ---")

    # Save data
    protected_save(df_simulation, sim_name, out_dir)
    protected_save(df_summary, summary_name, out_dir)

    # Directory message
    print(f"Files saved in: {os.getcwd()}")
    print("─" * 75)

#endregion

#region <def PLOT>

def filter_df(df, **kwargs):
    subset = df.copy()

    for col, val in kwargs.items():
        if col not in subset.columns:
            continue   # ← silently ignore non-columns
        subset = subset[subset[col] == val]

    return subset

def plot_param_sets(df, param_sets, 
                    confidence=True, 
                    out_dir="figures"):
    """
    param_sets : dict of lists
        Example:
        {
            "beta_0": [0, 1/3, 1, 3],
            "beta_1": [0],
            ...
            "incl_var": ["('T','M','C')"]
        }

    Always conditions on n and sd_Y to avoid mixing distributions.
    """

    print("--- Plot histograms ---")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # 1) Expand parameter grid (dict of lists → list of dicts)
    # --------------------------------------------------
    keys = list(param_sets.keys())
    values = list(param_sets.values())

    expanded_param_sets = [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]

    # --------------------------------------------------
    # 2) Background parameters (always fixed per plot)
    # --------------------------------------------------
    background_sets = (
        df[["n", "sd_Y"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )

    # --------------------------------------------------
    # 3) Plot
    # --------------------------------------------------
    for params in expanded_param_sets:
        for bg in background_sets:

            full_params = {**params, **bg}
            subset = filter_df(df, **full_params)

            if subset.empty:
                continue

            if "coef_T" not in subset.columns:
                print("❌ column 'coef_T' not in DataFrame.")
                return

            plt.figure(figsize=(10, 4))
            subset["coef_T"].hist(
                bins=50,
                density=True,
                alpha=0.4,
                color="cornflowerblue",
                edgecolor="black"
            )

            if confidence:
                ci_lower, ci_upper = np.percentile(
                    subset["coef_T"], [2.5, 97.5]
                )

                plt.axvline(ci_lower, color="gray", linestyle="--", lw=1,
                            label="95% CI")
                plt.axvline(ci_upper, color="gray", linestyle="--", lw=1)

            plt.axvline(
                subset["direct_effect"].iloc[0],
                color="blue",
                linestyle=":",
                lw=1.5,
                label=f"Direct ({subset['direct_effect'].iloc[0]:.2f})"
            )

            plt.axvline(
                subset["total_effect"].iloc[0],
                color="red",
                linestyle=":",
                lw=1.5,
                label=f"Total ({subset['total_effect'].iloc[0]:.2f})"
            )

            title_params = {
                ("ß_" + k[len("beta_"):] if k.startswith("beta_") else k):
                    (f"{v:.2f}" if isinstance(v, float) else v)
                for k, v in full_params.items()
            }

            beta_items = []
            other_items = []

            for k, v in title_params.items():
                if k.startswith("ß_"):
                    beta_items.append(f"{k}: {v}")
                else:
                    other_items.append(f"{k}: {v}")

            line1 = ", ".join(beta_items)
            line2 = ", ".join(other_items)

            plt.title(f"{line1}\n{line2}", pad=12)
            plt.xlabel("Estimated effect")
            plt.ylabel("Density")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)

            # --------------------------------------------------
            # 4) Safe filename (includes n & sd_Y)
            # --------------------------------------------------
            clean_name = "_".join(
                f"{('ß_' + k[len('beta_'):] if k.startswith('beta_') else k)}-{v:.2f}"
                if isinstance(v, float)
                else f"{('ß_' + k[len('beta_'):] if k.startswith('beta_') else k)}-{v}"
                for k, v in full_params.items()
            )

            plt.savefig(os.path.join(out_dir, f"{clean_name}.png"), dpi=300)
            plt.savefig(os.path.join(out_dir, f"{clean_name}.svg"))
            plt.close()

            print(f"✔ Saved: {clean_name}")

    print("─" * 75)









# def plot_param_sets(df, param_sets, out_dir="figures"):
#     """
#     For each parameter set:
#        1) filter df
#        2) plot histogram of coef_T only
#        3) save for each set
#     """

#     print("--- Plot histograms ---")

#     # Check if give directory existis, if not: make
#     os.makedirs(out_dir, exist_ok=True)

#     # Loop over selected parameters
#     for params in param_sets:

#         # Filter parameters
#         subset = filter_df(df, **params)

#         if subset.empty:
#             print(f"⚠️ No rows match filters: {params}")
#             continue

#         if "coef_T" not in subset.columns:
#             print("❌ column 'coef_T' not in DataFrame.")
#             return

#         ci_lower, ci_upper = np.percentile(subset['coef_T'], [2.5, 97.5])

#         formatted_params = {
#             k: (f"{v:.2f}" if isinstance(v, float) else v)
#             for k, v in params.items()
#         }

#         plt.figure(figsize=(10, 4))
#         subset["coef_T"].hist(bins=50, density=True, alpha=0.4, color='cornflowerblue', edgecolor='black')

#         plt.axvline(ci_lower, color='gray', linestyle='--', lw=1, label='95% CI bounds')
#         plt.axvline(ci_upper, color='gray', linestyle='--', lw=1)
#         plt.axvline(subset['direct_effect'].iloc[0], 
#                     color='blue', linestyle=':', lw=1.5, 
#                     label=f'Direct causal effect ({subset['direct_effect'].iloc[0].round(2)})')
#         plt.axvline(subset['total_effect'].iloc[0], 
#                     color='red', linestyle=':', lw=1.5, 
#                     label=f'Total causal effect ({subset['total_effect'].iloc[0].round(2)})')

#         plt.title(str(formatted_params))
#         plt.xlim(-5, 8)
#         plt.xticks(np.arange(-5, 9, 1))
#         plt.xlabel('Estimated effect')
#         plt.ylabel('Density')
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.6)

#         # Dynamic filename based on parameters
#         clean_name = "_".join(
#             f"{k}-{v:.2f}" if isinstance(v, float) else f"{k}-{v}"
#             for k, v in params.items()
#         )
#         png_path = os.path.join(out_dir, f"{clean_name}.png")
#         svg_path = os.path.join(out_dir, f"{clean_name}.svg")

#         # Save both formats
#         plt.savefig(png_path, dpi=300)          # PNG high-res
#         plt.savefig(svg_path)                   # SVG vector (no dpi needed)

#         plt.close()

#         print(f"✔ Saved: {png_path}")
#         print(f"✔ Saved: {svg_path}")
        
#     print("─" * 75)
