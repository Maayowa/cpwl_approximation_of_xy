import numpy as np
import matplotlib.pyplot as plt
import os
from ortools.math_opt.python import quadratic_constraints


def constraint_writer(model, solver, method):
    """Write constraint details to file based on solver type and method."""
    num_quadratic = model.mathopt_model.get_num_quadratic_constraints()
    os.makedirs("constraints_log", exist_ok=True)

    filename = f"constraints_log/{solver.name}_{method.value}_log.txt"
    # filename = f"constraints_log/{solver.name}_{method.value}.txt"
    with open(filename, "w") as f:
        f.write(f"Solver: {solver.name}, Method: {method.value}\n")
        f.write(f"Number of quadratic constraints: {num_quadratic}\n")

        for constraint in model.constraints:
            if isinstance(constraint, quadratic_constraints.QuadraticConstraint):
                expr = ""
                expr += f"{constraint.lower_bound:.2f} <= "
                for linear_term in constraint.linear_terms():
                    expr += f"{str(linear_term)} + "
                for quadratic_term in constraint.quadratic_terms():
                    expr += f"{str(quadratic_term)} + "
                f.write(f"\nQuadratic Constraint {constraint.id}:  {constraint.name}\n")
                f.write(
                    f" Expression: {expr[:-3].replace('1.0 * ', '')} <= {constraint.upper_bound:.2f}\n"
                )
            else:
                f.write(f"\nLinear Constraint:  {constraint.name}\n")
                f.write(
                    f"Linear constraint: {str(constraint.as_bounded_linear_expression()).replace('1.0 * ', '')}\n"
                )


def plot_bilinear_comparison(model, solve_result, inputs, solver, method, target_error):
    """
    Plot comparison of bilinear product approximation vs. latent variable.

    Args:
        model: Model object containing variables
        solve_result: Solution object from optimization
        inputs: Input data dictionary
        solver: Solver type
        method: Bilinear method used
        target_error: Target error value
    """
    # Configure plotting
    plt.rcParams["font.family"] = "Century Schoolbook"
    os.makedirs("output/figures", exist_ok=True)

    q_var = model.variables["q_gg"]
    t_var = model.variables["T_gg"]
    qt_var = model.variables["QT_latent"]

    nt = len(inputs["time_horizon"])
    nps = len(inputs["river_nodes"])

    # Extract variable values
    qt_latent = np.array(
        [
            [solve_result.variable_values()[qt_var[p][t]] for t in range(nt)]
            for p in range(nps)
        ]
    )
    q_val = np.array(
        [
            [solve_result.variable_values()[q_var[p][t]] for t in range(nt)]
            for p in range(nps)
        ]
    )
    t_val = np.array(
        [
            [solve_result.variable_values()[t_var[p][t]] for t in range(nt)]
            for p in range(nps)
        ]
    )

    # Compute product and plot
    qt_estimated = q_val * t_val

    plt.figure(figsize=(10, 6))
    plt.plot(qt_estimated[1], label="Direct Multiplication", linewidth=2)
    plt.plot(qt_latent[1], "--o", label="Latent Variable", markevery=12)
    plt.xlabel("Time (hrs)", fontsize=11)
    plt.ylabel("Flow (cms)", fontsize=11)
    plt.legend(fontsize=10)
    plt.title(
        f"Bilinear Approximation Comparison - {solver.name} - {method.value} (Log Encoding)",
        # f"Bilinear Approximation Comparison - {solver.name} - {method.value}",
        fontsize=12,
    )
    plt.tight_layout()

    filename = (
        f"output/figures/flowplot_log-{solver.name}-{method.value}-{target_error}.png"
        # f"output/figures/flowplot-{solver.name}-{method.value}-{target_error}.png"
    )
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Plot saved: {filename}")
