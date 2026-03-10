import random

import numpy as np
from process_bigraph import Process
from readdy import ReactionDiffusionSystem, Simulation
from simularium_readdy_models.actin import (
    ActinSimulation,
)
from simularium_readdy_models.common import ReaddyUtil

from pb_multiscale_actin.library.utils import get_monomers, simulate_readdy, id_difference, compare_particles, \
    pre_sim_particles, compare_topologies, pre_sim_topologies


class ReaddyActinMembrane(Process):
    """
    This process runs ReaDDy models with coarse-grained particle
    actin filaments and membrane patches.
    """

    config_schema = {
        "name": "string",
        "internal_timestep": "float",
        "box_size": "tuple[float,float,float]",
        "periodic_boundary": "boolean",
        "reaction_distance": "float",
        "n_cpu": "integer",
        "only_linear_actin_constraints": "boolean",
        "reactions": "boolean",
        "dimerize_rate": "float",
        "dimerize_reverse_rate": "float",
        "trimerize_rate": "float",
        "trimerize_reverse_rate": "float",
        "pointed_growth_ATP_rate": "float",
        "pointed_growth_ADP_rate": "float",
        "pointed_shrink_ATP_rate": "float",
        "pointed_shrink_ADP_rate": "float",
        "barbed_growth_ATP_rate": "float",
        "barbed_growth_ADP_rate": "float",
        "nucleate_ATP_rate": "float",
        "nucleate_ADP_rate": "float",
        "barbed_shrink_ATP_rate": "float",
        "barbed_shrink_ADP_rate": "float",
        "arp_bind_ATP_rate": "float",
        "arp_bind_ADP_rate": "float",
        "arp_unbind_ATP_rate": "float",
        "arp_unbind_ADP_rate": "float",
        "barbed_growth_branch_ATP_rate": "float",
        "barbed_growth_branch_ADP_rate": "float",
        "debranching_ATP_rate": "float",
        "debranching_ADP_rate": "float",
        "cap_bind_rate": "float",
        "cap_unbind_rate": "float",
        "hydrolysis_actin_rate": "float",
        "hydrolysis_arp_rate": "float",
        "nucleotide_exchange_actin_rate": "float",
        "nucleotide_exchange_arp_rate": "float",
        "verbose": "boolean",
        "use_box_actin": "boolean",
        "use_box_arp": "boolean",
        "use_box_cap": "boolean",
        "obstacle_radius": "float",
        "obstacle_diff_coeff": "float",
        "use_box_obstacle": "boolean",
        "position_obstacle_stride": "integer",
        "displace_pointed_end_tangent": "boolean",
        "displace_pointed_end_radial": "boolean",
        "tangent_displacement_nm": "float",
        "radial_displacement_radius_nm": "float",
        "radial_displacement_angle_deg": "float",
        "longitudinal_bonds": "boolean",
        "displace_stride": "integer",
        "bonds_force_multiplier": "float",
        "angles_force_constant": "float",
        "dihedrals_force_constant": "float",
        "actin_constraints": "boolean",
        "use_box_actin": "boolean",
        "actin_box_center_x": "float",
        "actin_box_center_y": "float",
        "actin_box_center_z": "float",
        "actin_box_size_x": "float",
        "actin_box_size_y": "float",
        "actin_box_size_z": "float",
        "add_extra_box": "boolean",
        "barbed_binding_site": "boolean",
        "binding_site_reaction_distance": "float",
        "add_membrane": "boolean",
        "membrane_center_x": "float",
        "membrane_center_y": "float",
        "membrane_center_z": "float",
        "membrane_size_x": "float",
        "membrane_size_y": "float",
        "membrane_size_z": "float",
        "membrane_particle_radius": "float",
        "obstacle_controlled_position_x": "float",
        "obstacle_controlled_position_y": "float",
        "obstacle_controlled_position_z": "float",
        "random_seed": "integer",
        "total_steps": "float",
        "actin_concentration": "integer",  # 0
        "arp23_concentration": "integer",  # 0
        "cap_concentration": "integer",  # 0
        "n_fixed_monomers_barbed": "integer",  # 0
        "tangent_displace_speed_um_s": "float",
        "plot_actin_compression": "boolean",  # true
        "visualize_edges": "boolean",  # true
        "visualize_normals": "boolean",  # true
        "visualize_control_pts": "boolean",  # true
    }

    def initialize(self, config, readdy_system=None):
        random.seed(self.config["random_seed"])
        np.random.seed(self.config["random_seed"])
        actin_simulation = ActinSimulation(self.config, False, False, readdy_system)
        self.readdy_system: ReactionDiffusionSystem = actin_simulation.system
        self.readdy_simulation: Simulation = actin_simulation.simulation

    def initial_state(self):
        return get_monomers()

    def inputs(self):
        return {
            "topologies": "map[overwrite[topology]]",
            "particles": "map[overwrite[particle]]",
        }

    def outputs(self):
        return {
            "topologies": "map[overwrite[topology]]",
            "particles": "map[overwrite[particle]]",
        }

    def update(self, inputs, interval: float):
        self.initialize(self.config, self.readdy_system)
        monomers = {
            'particles': inputs['particles'],
            'topologies': inputs['topologies'],
        }

        compare_particles(monomers['particles'], pre_sim_particles, False)
        compare_topologies(monomers['topologies'], pre_sim_topologies)

        ReaddyUtil.add_monomers_from_data(self.readdy_simulation, monomers)

        simulate_readdy(
            self.config["internal_timestep"],
            self.readdy_system,
            self.readdy_simulation,
            interval,
        )

        id_diff = id_difference(self.readdy_simulation.current_topologies)

        readdy_monomers = ReaddyUtil.get_current_monomers(
            self.readdy_simulation.current_topologies, id_diff
        )

        return readdy_monomers
