from typing import Any

from bigraph_schema import allocate_core, Core
from process_bigraph import Composite
from process_bigraph.emitter import emitter_from_wires, gather_emitter_results

from pb_multiscale_actin.processes import ReaddyActinMembrane, SimulariumEmitter


def get_default_config() -> dict[str, Any]:
    return {
        "name": "actin_membrane",
        "random_seed": 0,
    }

def register_items_into_core(core: Core):
    particle = {
        "type_name": "string",
        "position": "array",
        "neighbor_ids": "overwrite[list[integer]]",
        "unique_id": "maybe[integer]"
    }
    topology = {
        "type_name": "string",
        "particle_ids": "overwrite[list[integer]]",
    }
    core.register_type("topology", topology)
    core.register_type("particle", particle)

    core.register_link(
        "pb_multiscale_actin.processes.readdy_actin_membrane.ReaddyActinMembrane",
        ReaddyActinMembrane,
    )
    core.register_link(
        "pb_multiscale_actin.processes.simularium_emitter.SimulariumEmitter",
        SimulariumEmitter,
    )


def generate_readdy_pbg(output_dir):
    emitters_from_wires = emitter_from_wires(
        {
            "particles": ["particles"],
            "topologies": ["topologies"],
            "global_time": ["global_time"],
        },
        address="local:pb_multiscale_actin.processes.simularium_emitter.SimulariumEmitter",
    )
    emitters_from_wires["config"]["output_dir"] = output_dir

    state = {
        "emitter": emitters_from_wires,
        "readdy": {
            "_type": "process",
            "config": get_default_config(),
            "address": "local:pb_multiscale_actin.processes.readdy_actin_membrane.ReaddyActinMembrane",
            "inputs": {"particles": ["particles"], "topologies": ["topologies"]},
            "outputs": {"particles": ["particles"], "topologies": ["topologies"]},
        },
    }
    return state


def run_readdy_actin_membrane(total_time=3):
    state = generate_readdy_pbg(output_dir="")

    core = allocate_core()
    register_items_into_core(core)

    sim = Composite(
        {
            "state": state,
        },
        core=core,
    )

    # simulate
    sim.run(total_time)  # time in ns

    results = gather_emitter_results(sim)


if __name__ == "__main__":
    run_readdy_actin_membrane()
