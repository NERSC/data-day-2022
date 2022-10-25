# nersc-pegasus-example

There are a few peices of software that need to be setup for Pegaus workflows to work. We will go over how to download and use the software at NERSC.

## Setting up HTCondor

Download newsest pre-compiled version of HTCondor built for [Centos8](https://research.cs.wisc.edu/htcondor/tarball/current).

```bash
wget https://research.cs.wisc.edu/htcondor/tarball/current/9.11.2/release/condor-9.11.2-x86_64_CentOS8-stripped.tar.gz
tar -xvf condor-9.11.2-x86_64_CentOS8-stripped.tar.gz
export CONDOR_INSTALL=/path/to/condor
export PATH=${CONDOR_INSTALL}/bin:${CONDOR_INSTALL}/sbin:$PATH
```


## Setting up pegasus

Download newsest pre-compiled version of pegagsus built for [rhel8](http://download.pegasus.isi.edu/pegasus).

```bash
wget http://download.pegasus.isi.edu/pegasus/5.0.2/pegasus-binary-5.0.2-x86_64_rhel_8.tar.gz
tar -xvf pegasus-binary-5.0.2-x86_64_rhel_8.tar.gz
export PEGASUS_INSTALL=/path/to/pegasus
export PATH=${PEGASUS_INSTALL}/bin:$PATH
```

### Create conda environment

```bash
conda create -n pegasus python=3.10
conda activate pegasus
conda install -c conda-forge pegasus-wms
```

## Running HTCondor on the workflow qos

The current method for running HTCondor on perlmutter takes advantage of scrontab to run a long lasting job. To configure the job on perlmutter you can edit your scrontab with `scrontab -e` and place the folowing configuration in that file. This will start a new HTCondor server with a the scheduler running on the cron qos for longer running jobs.


```bash
#SCRON -q cron
#SCRON -A <account>
#SCRON -t 12:00:00
#SCRON -o output-%j.out
#SCRON --open-mode=append
*/10 * * * * /full/path/to/htcondor/start_htcondor.sh
```


# Starting a pegasus workflow

Once we have all the componenets needed we can start building our workflow in pegasus using the python api. There is an example workflow to split a file and count the lines in [workflow_generator.py](workflow_generator.py).