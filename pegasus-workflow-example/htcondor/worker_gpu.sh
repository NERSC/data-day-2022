#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -A <TODO: Add account>
#SBATCH -t 00:20:00
# #SBATCH --job-name=htcondor_worker
#SBATCH --exclusive

#SBATCH -e gpu_worker_%j.err
#SBATCH -o gpu_worker_%j.out

function killall {
    echo $(date)": Stopping workers"
    kill $(ps aux | grep -v grep | grep -i condor | awk '{print $2}')
    echo $(date)": Done!"
}

trap killall EXIT

# Move to the correct folder
cd ${HOME}/htcondor_workflow_scron

# For each node start a worker
echo $(date)": Starting Nodes "
## TODO: Change this to where you have placed the HTCondor install
export CONDOR_INSTALL=/path/to/condor
export PATH=$CONDOR_INSTALL/bin:$CONDOR_INSTALL/sbin:$PATH

export LOGDIR=${SCRATCH}/htcondorscratch

export CONDOR_SERVER=$(cat ${LOGDIR}/currenthost)

mkdir -p ${LOGDIR}/$(hostname)/log
mkdir -p ${LOGDIR}/$(hostname)/execute
mkdir -p ${LOGDIR}/$(hostname)/spool

export SCRIPTDIR="$(dirname -- "$BASH_SOURCE")"
export CONDOR_CONFIG=${SCRIPTDIR}/htcondor_worker.conf
echo $CONDOR_CONFIG

condor_master -f
