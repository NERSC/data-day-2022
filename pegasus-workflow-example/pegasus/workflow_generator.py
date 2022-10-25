#!/usr/bin/env python3
from Pegasus.api import (
    Properties,
    SiteCatalog,
    Site,
    Directory,
    FileServer,
    Operation,
    Grid,
    Scheduler,
    SupportedJobs,
    TransformationCatalog,
    Container,
    Transformation,
    ReplicaCatalog,
    Workflow,
    File,
    Job
)
import os
import logging
from pathlib import Path
from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API ------------------------------------------------------


class SplitWorkflow:

    # --- Init ----------------------------------------------------------------
    def __init__(self, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_name = "split"
        self.wf_dir = os.path.join("/pscratch/sd/t/tylern/pegasus")

    # --- Configuration (Pegasus Properties) ----------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()
        # Tells Pegasus to try and create symlinks for input files
        self.props["pegasus.transfer.links"] = "true"

        # Tells Pegasus to by the staging site ( creation of stage-in jobs) as
        # data is available directly on compute nodes
        self.props["pegasus.transfer.bypass.input.staging"] = "true"

    def write(self):
        # if not self.sc is None:
        #     self.sc.write()
        self.props.write()
        self.tc.write()
        self.rc.write()
        self.wf.write()

    # --- Site Catalog --------------------------------------------------------
    def create_sites_catalog(self, exec_site_name="perlmutter"):
        self.sc = SiteCatalog()

        shared_scratch_dir = os.path.join(
            "/pscratch/sd/t/tylern/pegasus/scratch")
        local_storage_dir = os.path.join(
            "/pscratch/sd/t/tylern/pegasus/output")

        local = Site("local").add_directories(
            Directory(Directory.SHARED_SCRATCH, shared_scratch_dir).add_file_servers(
                FileServer("file://" + shared_scratch_dir, Operation.ALL)
            ),
            Directory(Directory.LOCAL_STORAGE, local_storage_dir).add_file_servers(
                FileServer("file://" + local_storage_dir, Operation.ALL)
            ),
        )

        exec_site = Site("perlmutter")\
            .add_grids(
            Grid(grid_type=Grid.BATCH, scheduler_type=Scheduler.SLURM,
                 contact="${NERSC_USER}@saul-p1.nersc.gov", job_type=SupportedJobs.COMPUTE)
        )\
            .add_directories(
            Directory(
                Directory.SHARED_SCRATCH, shared_scratch_dir)
            .add_file_servers(FileServer("file:///pscratch/sd/t/tylern/pegasus/scratch", Operation.ALL)),
            Directory(
                Directory.SHARED_STORAGE, "/pscratch/sd/t/tylern/pegasus/storage")
            .add_file_servers(FileServer("file:///pscratch/sd/t/tylern/pegasus/storage", Operation.ALL))
        )\
            .add_env(key="PEGASUS_HOME", value="${NERSC_PEGASUS_HOME}")\
            .add_condor_profile(grid_resource='batch slurm')\
            .add_pegasus_profile(
            style="glite",
            data_configuration="sharedfs",
            project="nstaff",
            queue="debug",
            change_dir="true",
            create_dir="true",
            glite_arguments="-C cpu",
            cores="2",
            memory="4GB",
            runtime="1200",
            nodes=1,
        )

        self.sc.add_sites(local, exec_site)

    # --- Transformation Catalog (Executables and Containers) -----------------
    def create_transformation_catalog(self, exec_site_name="perlmutter"):
        self.tc = TransformationCatalog()

        # Create a container to run exes in
        ubuntu = Container(
            "ubuntu",
            Container.SHIFTER,
            image="shifter:///ubuntu:latest"
        )
        # Add it to the yml file
        self.tc.add_containers(ubuntu)

        # Create transforms or exes
        wc = Transformation(
            "wc", site=exec_site_name, pfn="/usr/bin/wc", is_stageable=False,
        )
        # The split command will be run in the container
        split = Transformation(
            "split", site=exec_site_name, pfn="/usr/bin/split", is_stageable=False,
            container=ubuntu
        )
        # Add the exes to the yml file
        self.tc.add_transformations(split, wc)

    # --- Replica Catalog ------------------------------------------------------
    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()

        # This is the input data we will be using
        self.rc.add_replica(
            "local", "test.csv", os.path.join(
                self.wf_dir, "input", "test.csv")
        )
        # Output data is added to the replica with
        # the register_replica options in the workflow section

    # --- Create Workflow -----------------------------------------------------
    def create_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        # Defines the test file
        test_file = File("test.csv")
        num_splits = 4
        # the split job that splits the test file into smaller chunks
        split = (
            Job("split")
            .add_args("-n", num_splits, "-d", "-a", 1, test_file, "part.")
            .add_inputs(test_file)
            .add_pegasus_profile(label="p1")
        )
        self.wf.add_jobs(split)

        # we do a parmeter sweep on the first 4 chunks created
        for c in range(num_splits):
            part = File("part.%s" % c)
            split.add_outputs(part, stage_out=True, register_replica=True)
            count = File("count.txt.%s" % c)
            wc = (
                Job("wc")
                .add_args("-l", part)
                .add_inputs(part)
                .set_stdout(count, stage_out=True, register_replica=True)
                .add_pegasus_profile(label="p1")
            )

            self.wf.add_jobs(wc)


if __name__ == "__main__":
    parser = ArgumentParser(description="Pegasus Split Workflow")

    parser.add_argument(
        "-s",
        "--sites_catalog",
        action="store_true",
        help="Skip site catalog creation",
        default=False
    )
    parser.add_argument(
        "-e",
        "--execution_site_name",
        metavar="STR",
        type=str,
        default="perlmutter",
        help="Execution site name (default: perlmutter)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="STR",
        type=str,
        default="workflow.yml",
        help="Output file (default: workflow.yml)",
    )

    args = parser.parse_args()

    workflow = SplitWorkflow(args.output)

    if args.sites_catalog:
        print("Creating execution sites...")
        workflow.create_sites_catalog(args.execution_site_name)

    print("Creating workflow properties...")
    workflow.create_pegasus_properties()

    print("Creating transformation catalog...")
    workflow.create_transformation_catalog(args.execution_site_name)

    print("Creating replica catalog...")
    workflow.create_replica_catalog()

    print("Creating split workflow dag...")
    workflow.create_workflow()

    workflow.write()
