# trinity run --config examples/async_gsm8k/explorer.yaml 2>&1 | tee explorer.log &
# sleep 30
# trinity run --config examples/async_gsm8k/trainer.yaml 2>&1 | tee trainer.log &

python -m trinity.cli.launcher run --config examples/async_gsm8k/explorer.yaml 2>&1 | tee explorer.log &
sleep 30
python -m trinity.cli.launcher run --config examples/async_gsm8k/trainer.yaml 2>&1 | tee trainer.log &
