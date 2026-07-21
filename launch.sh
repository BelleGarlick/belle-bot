export PYTHONPATH=.
export REPLAYS_PATH="replays"

python belle_bot/infra/fabric/service.py \
& python belle_bot/sensors/gps/service.py \
& python belle_bot/sensors/imu/service.py \
& python belle_bot/sensors/cameras/service.py \
& python belle_bot/sensors/microphone/service.py \
echo "Done"
