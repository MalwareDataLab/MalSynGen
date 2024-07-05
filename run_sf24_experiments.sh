#!/bin/bash

pipenv install -r requirements.txt

pipenv run python3 run_campaign.py -c SF24_4096_2048_10
