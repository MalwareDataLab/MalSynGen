#!/bin/bash

#pipenv install -r requirements.txt

pipenv run python3 run_campaign.py -c campanhas_SF24
if command -v jupyter &> /dev/null
then
    jupyter notebook plots.ipynb
else
   pip install notebook
   jupyter notebook plots.ipynb
fi

