import shutil
import os.path

import matsim.runtime.eqasim as eqasim

def configure(context):
    context.stage("matsim.simulation.prepare")

    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")

def execute(context):
    config_path = "%s/%s" % (
        context.path("matsim.simulation.prepare"),
        context.stage("matsim.simulation.prepare")
    )

    # Run routing
    output_dir = os.path.join(context.config("output_path"), "simulation")
    eqasim.run(context, "org.sutlab.hannover.RunSimulation", [
        "--config-path", config_path,
        "--config:controller.lastIteration", str(60),
        "--config:controller.writeEventsInterval", str(60),
        "--config:controller.writePlansInterval", str(60),
        "--config:controller.outputDirectory", output_dir,
    ])

    print("CONTEXT PATH: %s" % context.path())
    assert os.path.exists("%s/simulation_output/output_events.xml.gz" % context.path())
