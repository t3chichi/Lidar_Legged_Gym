LI_LEGGYM_ROOT=/home/t3chichi/Lidar_legged_gym

li_leggym() {
    conda activate li_leggym
    cd $LI_LEGGYM_ROOT
    pgrep -f "tensorboard.*$LI_LEGGYM_ROOT" >/dev/null || {
        gnome-terminal -- bash -c "tensorboard --logdir=$LI_LEGGYM_ROOT/legged_gym/logs --bind_all; exec bash" &
        disown
    }
}

# EL_4090 PD-GRU lidar tasks (ported from lab repo). play_laptop.py keeps the
# infinite viewer loop and resets policy state on episode boundaries.
el4090_lidar() {
    if [ $# -lt 2 ]; then
        echo "用法: el4090_lidar <load_run> <checkpoint>"
        return 1
    fi
    python $LI_LEGGYM_ROOT/legged_gym/legged_gym/scripts/play_laptop.py \
        --task=el4090_lidar --num_envs=16 \
        --load_run=$1 --checkpoint=$2
}

el4090_lidar_tripod2_low() {
    if [ $# -lt 2 ]; then
        echo "用法: el4090_lidar_tripod2_low <load_run> <checkpoint>"
        return 1
    fi
    python $LI_LEGGYM_ROOT/legged_gym/legged_gym/scripts/play_laptop.py \
        --task=el4090_lidar_tripod2_low --num_envs=16 \
        --load_run=$1 --checkpoint=$2
}

el4090_lidar_tripod2_low_avoid() {
    if [ $# -lt 2 ]; then
        echo "用法: el4090_lidar_tripod2_low_avoid <load_run> <checkpoint>"
        return 1
    fi
    python $LI_LEGGYM_ROOT/legged_gym/legged_gym/scripts/play_laptop.py \
        --task=el4090_lidar_tripod2_low_avoid --num_envs=16 \
        --load_run=$1 --checkpoint=$2
}
