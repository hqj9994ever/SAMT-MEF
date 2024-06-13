# SAMT-MEF
Official Code for: Qianjun Huang, Guanyao Wu, Zhiying Jiang, Wei Fan*, Bin Xua and Jinyuan Liu, **“Leveraging a Self-adaptive Mean Teacher Model for Semi-supervised Multi-Exposure Image Fusion”**.

## Set Up on Your Own Machine

### Virtual Environment

We strongly recommend that you use Conda as a package manager.

```shell
# create virtual environment
conda create -n SAMT-MEF python=3.8
conda activate SAMT-MEF
# select and install pytorch version yourself (Necessary & Important)
# install requirements package
pip install -r requirements.txt
```
### Train
```shell
# Initializes the self-adaptive set
python create_candidate.py

# Train: 
python train.py
```
### Test
```shell
# Test: 
python test.py
```
