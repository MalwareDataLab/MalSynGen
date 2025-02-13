#!/bin/bash

#pipenv install -r requirements.txt

python3 run_campaign.py -c  kronodroid_r_jbcs_2025,kronodroid_e_jbcs_2025,android_p_jbcs_2025,adroit_jbcs_2025,drebin_jbcs_2025,androcrawl_jbcs_2025
if command -v jupyter &> /dev/null
then
    jupyter notebook plots.ipynb
else
   pip install notebook
   jupyter notebook plots.ipynb
fi

