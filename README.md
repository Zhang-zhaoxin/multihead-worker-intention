First, create a <data> folder, and then place the worker intent dataset to be recognized into the data folder.
Then use extract_activity.py to create the dataset.  
The folder format is 
- data
  - worker_intent_dataset
    - train
    - val
    - test

use cluster_compute.py to extract video key frames, and then run run.py
