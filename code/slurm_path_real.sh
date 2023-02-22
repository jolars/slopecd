#!/bin/sh

#SBATCH -N 4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -t 24:00:00
#
#SBATCH -A lu2022-7-44
#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL
#
#SBATCH -J res_path
#SBATCH -o res_path_%j.out
#SBATCH -e res_path_%j.out
cat $0

declare -a data_sets=(\
    "bcTCGA" \
    "Rhee2006" \
    "rcv1.binary" \
    "news20.binary"
)

declare -a solvers=(\
    "hybrid_cd" \ 
    "fista" \
    "anderson" \
    "admm"
)

i=1

for data_set in ${data_sets[@]}
do
    for solver in ${solvers[@]}
    do
    file_name="results/path_real/${data_set}_${solver}.csv"
    if test -f "${file_name}"; then 
        echo "${data_set}/${solver} results already exist"
    else
        echo "running ${data_set}/${solver} experiment"
        srun -Q --exclusive --overlap -n 1 -N 1 \
            slurm_path_real_worker.sh $i $data_set $solver &> worker_${SLURM_JOB_ID}_${i}.out &
        i=$((i+1))
    fi
    sleep 1
    done
done

wait
