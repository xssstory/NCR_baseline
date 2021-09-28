# NCR baseline code

Baseline code for Native Chinese Reader (NCR) dataset

How to reproduce the baseline:

1. Install required packages 
```
pip install requirements.txt
```

2. Download data, create a folder called `rc_data` in the current directory and move `train_2.json`, `dev_2.json` and `test_2.json` into the folder, then run following codes for training
```
# To run a base pretrained model
python run_classifier.py --task_name RC --do_train --do_eval --data_dir . --max_seq_length 512 --train_batch_size 64 --eval_batch_size 512 --learning_rate 5e-6 --num_train_epochs 10 --output_dir mac_base --gradient_accumulation_steps 1 --local_rank -1 --init_checkpoint hfl/chinese-macbert-base

# To run a large pretrained model
python run_classifier.py --task_name RC --do_train --do_eval --data_dir . --max_seq_length 512 --train_batch_size 32 --eval_batch_size 256 --learning_rate 2e-6 --num_train_epochs 10 --output_dir mac_large --gradient_accumulation_steps 2 --local_rank -1 --init_checkpoint hfl/chinese-macbert-large
```

After finishing running, there would be 2 output files `dev_output.csv` and `test_output.csv` in your output folder.