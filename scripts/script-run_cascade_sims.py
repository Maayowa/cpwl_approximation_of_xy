# %% Import packages
import os
import polars as pl
import numpy as np
from cpwllib.hydrocascade import HydroModel, ModelConfig, Methods
from ortools.math_opt.python import mathopt
from cpwllib.utils import solve_result_gap
import cpwllib.hydrocascade.user as user
from cpwllib import DATA_DIR
from google.protobuf import text_format

# from helpers import constraint_writer

# %% Run sims

df_dicts = []

SOLVE_TIME_LIMIT = 3600
OPTIMALITY_GAP = 0.005  # 0.005, 0.0001
n_vals = np.array([1, 3, 5])
approx_errors = 1 / (
    16 * n_vals**2
)  # error values corresponding to n = [1, 3, 5] in paper

# Folder for saving logs
os.makedirs("output", exist_ok=True)
savefile_name = "xy_paper_cascade_results_run1"

for solver in [
    mathopt.SolverType.GUROBI,
    mathopt.SolverType.GSCIP,
    mathopt.SolverType.HIGHS,
]:
    for method in [
        Methods.SUM_OF_CONVEX,
        Methods.TRIANGLES,
        Methods.POLYGONS,
        Methods.QUADRATIC,
    ]:
        for target_error in approx_errors:
            if method == Methods.QUADRATIC and solver == mathopt.SolverType.HIGHS:
                continue

            if method == Methods.QUADRATIC:
                target_error = 0.000001

            # df_.write_parquet(f"sim_results_{solver}_{method}_{target_error}.parquet")
            if os.path.exists(
                f"output/cascade_sim_results_{solver}_{method}_{target_error}.parquet"
            ):
                print(
                    f"Skipping existing results for solver {solver.name} and method {method.value} with target error {target_error}"
                )
                df_ = pl.read_parquet(
                    f"output/cascade_sim_results_{solver}_{method}_{target_error}.parquet"
                )
                df_dicts.append(df_.to_dicts()[0])

                continue

            print(
                f"Running with solver {solver.name} and method {method.value} with target error {target_error}"
            )

            config = ModelConfig(
                name="CPWL Cascade Hydropower Case Study",
                solver_type=solver,
                bilinear_method=method,
                target_error=target_error,
            )

            model = HydroModel(config)

            # Load excel config file and NetworkX network
            excel_path = os.path.abspath(DATA_DIR / "cascade_hydro/cascade_input.xlsx")
            inputs = user.load_hydro_inputs_from_excel(excel_path)
            network_path = os.path.abspath(DATA_DIR / "cascade_hydro/cascade_net.yaml")
            inputs = user.load_cascade_network(network_path, inputs)

            # continue
            # model.populate_from_inputs(inputs)
            model.build_hydro_cascade_model(inputs)

            # Write constraints to file
            # constraint_writer(model, solver, method)

            integer_count = 0
            continuous_count = 0
            for var in model.mathopt_model.variables():
                if var.integer:
                    integer_count += 1
                else:
                    continuous_count += 1

            print(f"Model has {integer_count} integer variables")

            model_proto = model.mathopt_model.export_model()

            # Save to file (text format)
            with open(f"output/{solver}_{method}_{target_error}.txt", "w") as f:
                f.write(text_format.MessageToString(model_proto))

            solve_result = model.solve(
                time_limit=SOLVE_TIME_LIMIT, optimality_gap=OPTIMALITY_GAP
            )
            solved = (
                solve_result.termination.reason is mathopt.TerminationReason.FEASIBLE
                or solve_result.termination.reason is mathopt.TerminationReason.OPTIMAL
            )

            df_dicts.append(
                {
                    "solver": solver.name,
                    "method": method.value,
                    "status": solve_result == None,
                    "target_error": target_error,
                    "runtime": solve_result.solve_time() if solve_result else None,
                    "variables": model.mathopt_model.get_num_variables(),
                    "integer_variables": integer_count,
                    "continuous_variables": continuous_count,
                    "linear_constraints": model.mathopt_model.get_num_linear_constraints(),
                    "quadratic_constraints": model.mathopt_model.get_num_quadratic_constraints(),
                    "node_count": (
                        solve_result.solve_stats.node_count if solve_result else None
                    ),
                    "simplex_iterations": (
                        solve_result.solve_stats.simplex_iterations
                        if solve_result
                        else None
                    ),
                    "objective_value": (
                        solve_result.objective_value() if solved else None
                    ),
                    "best_bound": (
                        solve_result.best_objective_bound() if solve_result else None
                    ),
                }
            )

            df_ = pl.DataFrame(df_dicts[-1])

            df_.write_parquet(
                f"output/cascade_sim_results_{solver}_{method}_{target_error}.parquet"
            )
            if method == Methods.QUADRATIC:
                break

df = pl.DataFrame(df_dicts)
df.write_parquet(f"{savefile_name}.parquet")
df.to_pandas().to_csv(f"{savefile_name}.csv", index=False)
print(df)
