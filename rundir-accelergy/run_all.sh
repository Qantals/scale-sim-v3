#!/bin/bash

########### User Input ##########################

while getopts c:t:p:o:i:s: flag
do
    case "${flag}" in
        c) scsimCfg=${OPTARG};;
        t) scsimTplg=${OPTARG};;
        p) scsimOutput=${OPTARG};;        
        o) allOutput=${OPTARG};;
        i) topoOption=${OPTARG};;
        s) intervalSize=${OPTARG};;
    esac
done 

if [[ $scsimCfg == ""  ||  $scsimTplg == ""  || $allOutput == "" ]]; then
 echo "Not enough input files privoded"
 echo "./run_all.sh -c <path_to_config_file> -t <path_to_topology_file> -p <path_to_scale-sim_log_dir> -o <path_to_final_output_dir> [-s <interval_size>]"
 exit 0
fi

# Default interval size if not specified
if [[ -z "$intervalSize" ]]; then
    intervalSize=1000
fi

echo "config file: $scsimCfg";
echo "topology file: $scsimTplg";
echo "scsim log dir: $scsimOutput";
echo "output dir: $allOutput";
echo "topology option: $topoOption";
echo "interval size: $intervalSize";

################################################

# Ensure log and output directories exist before resolving absolute paths
if [[ -n "$scsimOutput" ]]; then
    mkdir -p "$scsimOutput" || { echo "Failed to create scsim log dir at $scsimOutput"; exit 1; }
fi

if [[ -n "$allOutput" ]]; then
    mkdir -p "$allOutput" || { echo "Failed to create output dir at $allOutput"; exit 1; }
fi

scsimCfg=$(realpath "$scsimCfg")
scsimTplg=$(realpath "$scsimTplg")
scsimOutput=$(realpath "$scsimOutput")
allOutput=$(realpath "$allOutput")

rm -f accelergy_input/*.yaml

# Generate Accelergy::architecture.yaml from ScaleSim::scale.cfg
python3 preprocess.py -c $scsimCfg -t $scsimTplg -p $scsimOutput -o $allOutput

# Run Scale-sim
cd ..
# Build the scale.py command with optional -i flag
SCSIM_CMD="python3 scale.py -c $scsimCfg -t $scsimTplg -p $scsimOutput"
if [[ -n "$topoOption" ]]; then
    SCSIM_CMD="$SCSIM_CMD -i $topoOption"
fi
echo "Running: $SCSIM_CMD"
$SCSIM_CMD

# Extract Accelergy::action_count.yaml from ScaleSim::reuslts
cd rundir-accelergy
./create_action_count.sh


# Run Accelergy
./run_accelergy.sh

# ------------------------------------------------------------------
# Step 5: Interval-based transient energy estimation
# ------------------------------------------------------------------
echo ""
echo "--- Transient interval energy estimation ---"

# Extract run_name from config
run_name=$(grep -E '^run_name[[:space:]]*[=:]' "$scsimCfg" \
           | head -1 | sed 's/.*[=:] *//')

echo "run_name: $run_name"
echo "interval: $intervalSize cycles"

conda run -n scalesim python3 create_action_count_interval.py \
    --saved_folder "$allOutput" \
    --run_name "scale_sim_output_${run_name}" \
    --config "$scsimCfg" \
    --interval_size "$intervalSize"

echo ""
echo "Transient energy output:"
echo "  $allOutput/scale_sim_output_${run_name}/interval_energy.csv"
echo "  $allOutput/scale_sim_output_${run_name}/interval_activity.csv"
echo ""

# Generate power-over-time plots
echo "--- Generating power plots ---"
mkdir -p "$allOutput/plots"
conda run -n scalesim python3 plot_power_timeline.py \
    --input "$allOutput/scale_sim_output_${run_name}/interval_energy.csv" \
    --output "$allOutput/plots" \
    --freq_mhz 1000

echo "Power plots saved to: $allOutput/plots/"

# Post-Process
# gen_plot.ipynb
