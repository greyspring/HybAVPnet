# HybAVPnet:a novel hybrid network architec ture for antiviral peptide identification.


## Prerequisite
* Python3.8 keras2.2.4  scikit-learn 0.24.1 tensorflow 2.3.0 lightgbm 3.2.1


#### To train the model
```
python Model_Train.py

```
if you want to trian your own model, you can change the filepath in the Model_Train.py
##### To evaluate the performance on validation set
```
python Model_infer.py

```
if you want to predict on your dataset with pretrained model, you can change the filepath in Model_infer.py  
