#!/bin/bash

cd pytracking
mkdir -p pytracking/networks
python3 -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python3 -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
bash pytracking/utils/gdrive_download 1qgachgqks2UGjKx-GdO1qylBDdB1f9KN pytracking/networks/dimp50.pth
bash pytracking/utils/gdrive_download 1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk pytracking/networks/dimp18.pth
bash pytracking/utils/gdrive_download 1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU pytracking/networks/atom_default.pth
bash pytracking/utils/gdrive_download 1aWC4waLv_te-BULoy0k-n_zS-ONms21S pytracking/networks/resnet18_vggmconv1.pth