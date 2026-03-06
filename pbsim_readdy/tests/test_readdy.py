import math
import random

import numpy as np
from bigraph_schema import allocate_core
from process_bigraph import Composite
from readdy import ReactionDiffusionSystem, Simulation
from simularium_readdy_models import ReaddyUtil
from simularium_readdy_models.actin import ActinSimulation

from pb_multiscale_actin.create_readdy_pbif import generate_readdy_pbg, register_items_into_core, get_default_config
from pb_multiscale_actin.library.utils import simulate_readdy, id_difference, get_monomers, compare_particles, \
    compare_topologies, pre_sim_particles, pre_sim_topologies, expected_particles, expected_topologies


def test_readdy_actin_model() -> None:
    random.seed(0)
    np.random.seed(0)
    monomers = get_monomers()

    config = get_default_config()
    actin_sim = ActinSimulation(config, False, False)
    readdy_system: ReactionDiffusionSystem = actin_sim.system
    readdy_simulation: Simulation = actin_sim.simulation

    compare_particles(monomers['particles'], pre_sim_particles, False)
    compare_topologies(monomers['topologies'], pre_sim_topologies)

    ReaddyUtil.add_monomers_from_data(readdy_simulation, monomers)
    simulate_readdy(0.1, readdy_system=readdy_system, readdy_simulation=readdy_simulation, timestep=1)

    id_diff = id_difference(readdy_simulation.current_topologies)
    result_monomers = ReaddyUtil.get_current_monomers(
        readdy_simulation.current_topologies,
        id_diff
    )

    compare_particles(result_monomers["particles"], expected_particles)
    compare_topologies(result_monomers["topologies"], expected_topologies)


def test_readdy_actin_pb() -> None:
    state = generate_readdy_pbg(output_dir="")

    core = allocate_core()
    register_items_into_core(core)

    sim = Composite(
        {
            "state": state,
        },
        core=core,
    )

    compare_particles(sim.state["particles"], pre_sim_particles, False)
    compare_topologies(sim.state["topologies"], pre_sim_topologies)
    # simulate
    sim.run(1)  # time in ns

    compare_particles(sim.state["particles"], expected_particles)
    compare_topologies(sim.state["topologies"], expected_topologies)

