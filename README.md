First, create a <data> folder, and then place the worker intent dataset to be recognized into the data folder.
然后使用extract_activity.py 创建数据集。
文件夹格式为  
- data
 - activity
    - train
    - val
    - test

use  
cluster_compute.py 
to extract video key frames, and then  
run run.py
