#!/bin/sh

#SBATCH -N 2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -t 03:00:00
#
#SBATCH -A lu2022-7-44
#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL
#
#SBATCH -J res_path_simulated
#SBATCH -o res_path_simulated_%j.out
#SBATCH -e res_path_simulated_%j.out
cat $0

declare -a scenarios=(\
    "1" \
    "2" \
    "3"
)

declare -a solvers=(\
    "hybrid_cd" \ 
    "fista" \
    "anderson" \
    "admm"
)

i=1

for scenario in ${scenarios[@]}
do
    for solver in ${solvers[@]}
    do
    file_name="results/path_simulated/scenario${scenario}_${solver}.csv"
    if test -f "${file_name}"; then 
        echo "scenario${scenario}/${solver} results already exist"
    else
        echo "running scenario${scenario}/${solver} experiment"
        srun -Q --exclusive --overlap -n 1 -N 1 \
            slurm_path_simulated_worker.sh $i $scenario $solver &> worker_${SLURM_JOB_ID}_${i}.out &
        i=$((i+1))
    fi
    sleep 1
    done
done

wait
