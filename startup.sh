LI_LEGGYM_ROOT=/home/t3chichi/Lidar_legged_gym

li_leggym() {
    conda activate li_leggym
    cd $LI_LEGGYM_ROOT
    pgrep -f "tensorboard.*$LI_LEGGYM_ROOT" >/dev/null || {
        gnome-terminal -- bash -c "tensorboard --logdir=$LI_LEGGYM_ROOT/legged_gym/logs --bind_all; exec bash" &
        disown
    }
}

go2_lidar() {
    if [ $# -lt 2 ]; then
        echo "用法: go2_lidar <load_run> <checkpoint>"
        return 1
    fi
    python $LI_LEGGYM_ROOT/legged_gym/legged_gym/scripts/play_laptop.py \
        --task=go2_lidar_pd_risknet --num_envs=16 \
        --load_run=$1 --checkpoint=$2
}

go2_pillar() {
    if [ $# -lt 2 ]; then
        echo "用法: go2_pillar <load_run> <checkpoint>"
        return 1
    fi
    python $LI_LEGGYM_ROOT/legged_gym/legged_gym/scripts/play_laptop.py \
        --task=go2_lidar_pillar --num_envs=16 \
        --load_run=$1 --checkpoint=$2
}

go2_pretrain() {
    if [ $# -lt 2 ]; then
        echo "用法: go2_pretrain <load_run> <checkpoint>"
        return 1
    fi
    python $LI_LEGGYM_ROOT/legged_gym/legged_gym/scripts/play.py \
        --task=go2_pd_pretrain --num_envs=16 \
        --load_run=$1 --checkpoint=$2
}

go2_soft_pretrain() {
    if [ $# -lt 2 ]; then
        echo "用法: go2_soft_pretrain <load_run> <checkpoint>"
        return 1
    fi
    python $LI_LEGGYM_ROOT/legged_gym/legged_gym/scripts/play.py \
        --task=go2_soft_pretrain --num_envs=16 \
        --load_run=$1 --checkpoint=$2
}
