export SCENE="scene_1"

for SCENE in $SCENE;
do
    python3 render_experimental.py --model_path "output/synthetic/${SCENE}"  --configs arguments/mdnerf-dataset/cube.py --view_skip 5 --time_skip 1 --scale 0.5 --skip_video \
    --show_flow --flow_skip 40 --tracking_window 60 --log_deform 
done