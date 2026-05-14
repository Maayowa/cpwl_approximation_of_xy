import datetime
import logging
from tqdm import tqdm
from ortools.math_opt.python import mathopt, result
from cpwllib.xy_plot_and_constraints import MILP_or_QP_variables_and_constraints_hydro
from dataclasses import dataclass
import enum
import numpy as np


class Methods(enum.Enum):
    TRIANGLES = "triangles"
    POLYGONS = "polygons"
    SUM_OF_CONVEX = "sum of convex"
    QUADRATIC = "quadratic"


@dataclass
class ModelConfig:
    name: str
    solver_type: mathopt.SolverType
    bilinear_method: Methods = Methods.TRIANGLES
    target_error: float = 0.05
    # product_cpwl_method: ProductCPWLMethod = ProductCPWLMethod.SINGLE
    # product_linearization: ProductLinearizationConfig


class HydroModel:
    """
    Base class for a model
    """

    def __init__(self, config: ModelConfig):
        self.model_name = config.name
        self.logger = logging.getLogger(f"Model.{self.model_name}")
        self.parameters = {}
        self.variables = {}
        self.config = config

        pass

    def build_hydro_cascade_model(
        self,
        case_study,
        logarithmic_encoding=False,
    ):
        partition_method = self.config.bilinear_method.value
        target_error = self.config.target_error
        quadratic = partition_method == "quadratic"

        cfs_to_m3s = 0.028316847
        g = 9.8
        ft_to_m = 0.3048

        pricing = case_study["pricing"]
        elevation = case_study["elevation"]
        System = case_study["network"]
        model_init = case_study["config"]

        HPPs = list(System.nodes())
        # Use plant representation for now (1 unit)
        HPPs = {p: [0] for p in HPPs}

        model = mathopt.Model(name="Cascading")

        # sets
        N_timesteps = len(pricing)  # pricing by timestamps
        timesteps = range(N_timesteps)
        timesteps_1 = range(1, N_timesteps)

        # factor to rescale the model
        flow_factor = 1e-3
        head_factor = 1e-2
        power_factor = 1e-3

        # variables
        # unit level
        unit_power_release_AF = {
            p: [
                [
                    model.add_variable(
                        lb=0.0, name=f"unit_power_release_AF({p},{u},{t})"
                    )
                    for t in timesteps
                ]
                for u in HPPs[p]
            ]
            for p in HPPs
        }
        unit_power_generation_MW = {
            p: [
                [
                    model.add_variable(
                        lb=0.0, name=f"unit_power_generation_MW({p},{u},{t})"
                    )
                    for t in timesteps
                ]
                for u in HPPs[p]
            ]
            for p in HPPs
        }

        plant_power_release_AF = {
            p: [
                model.add_variable(lb=0.0, name=f"plant_power_release_AF({p},{t})")
                for t in timesteps
            ]
            for p in HPPs
        }
        plant_non_power_release_AF = {
            p: [
                model.add_variable(
                    lb=0.0,
                    ub=0.0,  # FIX TO ZERO FOR NOW
                    name=f"plant_non_power_release_AF({p},{t})",
                )
                for t in timesteps
            ]
            for p in HPPs
        }
        plant_total_release_AF = {
            p: [
                model.add_variable(lb=0.0, name=f"plant_total_release_AF({p},{t})")
                for t in timesteps
            ]
            for p in HPPs
        }
        plant_storage_volume_AF = {
            p: [
                model.add_variable(lb=0.0, name=f"plant_storage_volume_AF({p},{t})")
                for t in timesteps
            ]
            for p in HPPs
        }
        plant_forebay_elevation_ft = {
            p: [
                model.add_variable(lb=0.0, name=f"plant_forebay_elevation_ft({p},{t})")
                for t in timesteps
            ]
            for p in HPPs
        }
        plant_head_ft = {
            p: [
                model.add_variable(lb=0.0, name=f"plant_head_ft({p},{t})")
                for t in timesteps
            ]
            for p in HPPs
        }

        StoE_piece_choice = {p: [[] for t in timesteps] for p in HPPs}
        StoE_piece_storage = {p: [[] for t in timesteps] for p in HPPs}
        StoE_piece_elevation = {p: [[] for t in timesteps] for p in HPPs}

        # objective function

        model.maximize(
            sum(
                [
                    pricing.loc[t, p] * unit_power_generation_MW[p][u][t]
                    for p in HPPs
                    for u in HPPs[p]
                    for t in timesteps
                ]
            )
        )

        # total and power water release
        for p in HPPs:
            for t in timesteps:
                model.add_linear_constraint(
                    plant_total_release_AF[p][t]
                    == plant_power_release_AF[p][t] + plant_non_power_release_AF[p][t]
                )
                model.add_linear_constraint(
                    plant_power_release_AF[p][t]
                    == sum([unit_power_release_AF[p][u][t] for u in HPPs[p]])
                )

        # we assume the limit on power release, not power capacity
        for p in HPPs:
            for u in HPPs[p]:
                for t in timesteps:
                    unit_power_release_AF[p][u][t].upper_bound = (
                        model_init.loc[p, "max turbine release"] * flow_factor
                    )

        # link between head and forebay elevation
        for p in HPPs:
            # for the lowest reservoir, use tailwater
            if not list(System.successors(p)):
                for t in timesteps:
                    # define upper bound head
                    plant_head_ft[p][t].upper_bound = (
                        np.max(model_init.loc[p, "max elevation"])
                        - model_init.loc[p, "tailwater elevation"]
                    ) * head_factor
                    model.add_linear_constraint(
                        plant_head_ft[p][t]
                        == plant_forebay_elevation_ft[p][t]
                        - model_init.loc[p, "tailwater elevation"] * head_factor
                    )
            # coupling constraint
            else:
                pp = list(System.successors(p))[0]
                for t in timesteps:
                    # define upper bound head
                    plant_head_ft[p][t].upper_bound = (
                        np.max(model_init.loc[p, "max elevation"])
                        - np.min(model_init.loc[pp, "min elevation"])
                    ) * head_factor
                    model.add_linear_constraint(
                        plant_head_ft[p][t]
                        == plant_forebay_elevation_ft[p][t]
                        - plant_forebay_elevation_ft[pp][t]
                    )

        # elevation limits
        for p in HPPs:
            elevation_min = model_init.loc[p, "min elevation"]
            elevation_max = model_init.loc[p, "max elevation"]
            if isinstance(elevation_min, (float, int)):
                elevation_min = [elevation_min] * N_timesteps
            if isinstance(elevation_max, (float, int)):
                elevation_max = [elevation_max] * N_timesteps
            for t in timesteps:
                plant_forebay_elevation_ft[p][t].lower_bound = (
                    elevation_min[t] * head_factor
                )
                plant_forebay_elevation_ft[p][t].upper_bound = (
                    elevation_max[t] * head_factor
                )

        # water balance equations
        for p in HPPs:
            # t=0
            model.add_linear_constraint(
                plant_storage_volume_AF[p][t]
                - model_init.loc[p, "initial storage volume"] * flow_factor
                == model_init.loc[p, "side inflow"] * flow_factor
                - plant_total_release_AF[p][t]
                + sum([plant_total_release_AF[pp][t] for pp in System.predecessors(p)])
            )
            # t>0
            for t in timesteps_1:
                model.add_linear_constraint(
                    plant_storage_volume_AF[p][t] - plant_storage_volume_AF[p][t - 1]
                    == model_init.loc[p, "side inflow"] * flow_factor
                    - plant_total_release_AF[p][t]
                    + sum(
                        [plant_total_release_AF[pp][t] for pp in System.predecessors(p)]
                    )
                )

        # forebay as a function of storage volume...
        for p in HPPs:
            # cpwl_data = data_reservoir[p]["storage to elevation"]
            cpwl_data = elevation[elevation.Type == p].values
            N_pieces = cpwl_data.shape[0]
            for t in timesteps:
                # add variables
                for k in range(N_pieces):
                    StoE_piece_choice[p][t].append(
                        model.add_binary_variable(
                            name=f"StoE_piece_choice({p},{t},{k})"
                        )
                    )
                    StoE_piece_storage[p][t].append(
                        model.add_variable(
                            lb=0.0, name=f"StoE_piece_storage({p},{t},{k})"
                        )
                    )
                    StoE_piece_elevation[p][t].append(
                        model.add_variable(
                            lb=0.0, name=f"StoE_piece_elevation({p},{t},{k})"
                        )
                    )
                # add pwl constraints
                # elevation decomposition
                model.add_linear_constraint(
                    plant_forebay_elevation_ft[p][t]
                    == sum([StoE_piece_elevation[p][t][k] for k in range(N_pieces)])
                )
                # storage decomposition
                model.add_linear_constraint(
                    plant_storage_volume_AF[p][t]
                    == sum([StoE_piece_storage[p][t][k] for k in range(N_pieces)])
                )
                # one piece at a time
                model.add_linear_constraint(
                    sum([StoE_piece_choice[p][t][k] for k in range(N_pieces)]) == 1.0
                )
                for k in range(N_pieces):
                    # bounds of each storage component
                    model.add_linear_constraint(
                        StoE_piece_storage[p][t][k]
                        >= StoE_piece_choice[p][t][k] * cpwl_data[k, 0] * flow_factor
                    )
                    model.add_linear_constraint(
                        StoE_piece_storage[p][t][k]
                        <= StoE_piece_choice[p][t][k] * cpwl_data[k, 1] * flow_factor
                    )
                    # linear equation
                    model.add_linear_constraint(
                        StoE_piece_elevation[p][t][k]
                        == (head_factor / flow_factor)
                        * cpwl_data[k, 2]
                        * StoE_piece_storage[p][t][k]
                        + head_factor * cpwl_data[k, 3] * StoE_piece_choice[p][t][k]
                    )

        # unit_power_generation_MW as a function of unit_release and head (actually average of head)

        Qh_variables = MILP_or_QP_variables_and_constraints_hydro(
            model,
            unit_power_release_AF,
            plant_head_ft,
            quadratic=quadratic,
            target_error=target_error,
            partition_method=partition_method,
            logarithmic_encoding=logarithmic_encoding,
        )

        # Note HIGHS find solution with gap <2% after 641s under 168 hours, 'sum of convex' with no encoding (but still at 1.45% after 1,800s)
        # <2% and <1% after 371s 'polygons' with no encoding
        # <2% and <1% after 634s 'triangles' with no encoding

        Z = Qh_variables["Z"]

        # to convert the Q and h (AF/hr and ft) to MW
        # efficiency = {"BM": 0.85, "MP": 0.85, "CY": 0.42}
        efficiency = model_init["efficiency"]
        Qh_to_MW = (
            cfs_to_m3s
            * 12.1
            * ft_to_m
            * g
            * 1e-3
            * power_factor
            / (flow_factor * head_factor)
        )

        for p in HPPs:
            for u in HPPs[p]:
                for t in timesteps:
                    model.add_linear_constraint(
                        unit_power_generation_MW[p][u][t]
                        == efficiency[p] * Qh_to_MW * Z[p][u][t]
                    )

        # return model
        # self.variables = variables
        # self.constraints = constraints
        # self.constraint_id_main = constraint_id_main
        # self.constraint_id_unit = constraint_id_unit
        # self.constraint_id_time = constraint_id_time
        # self.parameters = parameters
        self.mathopt_model = model

    def solve(self, time_limit=10, optimality_gap=0.01) -> result.SolveResult:
        time_limit = datetime.timedelta(seconds=time_limit)
        params = mathopt.SolveParameters(
            enable_output=True,
            time_limit=time_limit,
            relative_gap_tolerance=optimality_gap,
        )

        return mathopt.solve(
            opt_model=self.mathopt_model,
            solver_type=self.config.solver_type,
            params=params,
        )
