# LM-TOAST
Code and data of the Findings of ACL 2023 [[paper](https://aclanthology.org/2023.findings-acl.624/)] "Making Pre-trained Language Models both Task-solvers and Self-calibrators".

# Installation

```sh
pip install -r requirements.txt
mkdir data
```



# Data Preparation

We use the same datasets and preprocessing scripts as in this [[paper](https://arxiv.org/abs/2211.00151)].
You can download the datasets from Google Drive [[download]](https://drive.google.com/file/d/1738RctASgLd-vRIGxo4ytZFR3Kpb5nU0/view?usp=share_link "downlaod datasets from Google Drive"), 
and upload the folder (**TextClassification**) to the `data` directory. Then, all the datasets used in the paper can be find in ./data/TextClassification.




# Experiments
```sh
export PYTHONPATH='pwd':$PYTHONPATH
python src/scripts/run.py --model_name t5 --scale base --dataset_name amazon_food --save_path amazon_t5base.ckpt
```






# Process Results
To compute the metrics for calibration. Run:
```sh
export PYTHONPATH='pwd':$PYTHONPATH
python src/scripts/metric.py --setting_list SETTING_LIST --model_list MODEL_LIST --dataset_list DATASET_LIST
```
By passing `SETTING_LIST`, `MODEL_LIST` and `DATASET_LIST`, you can find the final metrics for all the experiments in the directory `./metrics`.





# Citation
Please kindly cite our paper:
```
@inproceedings{chen-etal-2023-making,
    title = "Making Pre-trained Language Models both Task-solvers and Self-calibrators",
    author = "Chen, Yangyi  and
      Wang, Xingyao  and
      Ji, Heng",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    publisher = "Association for Computational Linguistics",    
}
```





