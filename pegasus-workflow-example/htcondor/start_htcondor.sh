#!/usr/bin/env bash

## TODO: Change this to where you have placed the HTCondor install
export CONDOR_INSTALL=/path/to/condor

# Gets directory the script is being run from
export SCRIPTDIR="$(dirname -- "$BASH_SOURCE")"

# Makes some directories to hold the HTCondor logs
export LOGDIR=${SCRATCH}/htcondorscratch
mkdir -p $LOGDIR/$(hostname)/log
mkdir -p $LOGDIR/$(hostname)/execute
mkdir -p $LOGDIR/$(hostname)/spool

# Works as a dns to make sure we can connect to the htcondor service after it starts
echo $(hostname) >$LOGDIR/currenthost
export CONDOR_SERVER=$(cat ${LOGDIR}/currenthost)

# Choose a random part number for HTCondor to use, HTCondor defaults to 9816
export CONDOR_PORT=9618

# Write a password to this file before starting
export PASSWORDFILE=${HOME}/.condor/cron.password
export PATH=${PATH}:${CONDOR_INSTALL}/bin:${CONDOR_INSTALL}/sbin

export CONDOR_CONFIG=${SCRIPTDIR}/htcondor_server.conf

condor_master -f
