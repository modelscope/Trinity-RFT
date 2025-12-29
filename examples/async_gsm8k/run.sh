#!/bin/bash
trinity run --config examples/async_gsm8k/trainer.yaml 2>&1 | tee trainer.log &
sleep 30
trinity run --config examples/async_gsm8k/explorer.yaml 2>&1 | tee explorer.log &
