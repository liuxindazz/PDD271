# PDD271
This is a Python3 / Pytorch implementation of PDD271, as described in the following paper:

**Plant Disease Recognition:A Large-Scale Benchmark Dataset and a Visual Region and Loss Reweighting Approach**, by
Xinda Liu, Weiqing Min, Shuhuan Mei, Lili Wang, and Shuqiang Jiang

To facilitate the plant disease recognition research, we construct a new large-scale plant disease dataset with 271 plant disease
categories and 220,592 images. Based on this dataset, we tackle plant disease recognition via reweighting both visual regions
and loss to emphasize diseased parts.
![avatar](https://github.com/liuxindazz/PDD271/raw/main/datasetShow.png)
Disease leaf image samples from various categories of PDD271 (one samples per category). The dataset contains three macro-classes:
Fruit Tree, Vegetable, and Field Crops.

## Setup

To run this code you need the following: 
+ a machine with multiple GPUs
+ Python3
+ other packages:   
`pip install -r requirements.txt`

## Testing the model

Use the `test.py` script to test the pretrained model.    
  `python3 test.py`
