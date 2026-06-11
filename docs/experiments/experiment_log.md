# Experiment Log

## EXP-001
Date: 11/6/2026
Objective: Establish a strong centralized baseline on the same COVID dataset used by the 2025 paper before introducing federated learning, encryption, or blockchain.
Dataset: COVID-19 Radiography Database
Hyperparameters:Model - EfficientNet-B0
                Initialization - ImageNet Pretrained
                Framework -	PyTorch
                Device - RTX 3060 Laptop GPU
                Batch Size - 16
                Epochs - 5
                Optimizer - AdamW
                Learning Rate -	1×10⁻⁴
                Input Size - 224×224
                Loss Function -	CrossEntropyLoss

Results:Epoch	Train Loss	Test Accuracy	F1 Score
            1	0.3899	    96.00%	    96.00%
            2	0.1706	    96.75%	    96.75%
            3	0.1045	    97.50%	    97.50%
            4	0.0874	    98.00%	    98.00%
            5	0.0724	    95.25%	    95.24%
Observations: EfficientNet-B0 achieved its highest performance at Epoch 4, obtaining 98.00% accuracy and F1-score on the COVID-vs-Normal classification task. Performance degradation observed at Epoch 5 suggests early signs of overfitting despite the relatively small dataset.


## EXP-002
Date: 11/6/2026
Objective: Evaluate the effect of standard FedAvg on COVID classification using EfficientNet-B0.
Dataset: COVID-19 Radiography Database
Hyperparameters:Clients - 4
                Local Epochs - 	1
                Global Rounds -	5
                Model -	EfficientNet-B0
                Aggregation - FedAvg
                Dataset - COVID Binary
                Device - RTX 3060

Results:Round	Accuracy	F1	Comm Cost	Time
            1	81.00%	80.62%	123.66 MB	43.83s
            2	89.75%	89.74%	123.66 MB	43.78s
            3	93.50%	93.48%	123.66 MB	50.62s
            4	95.50%	95.50%	123.66 MB	50.74s
            5	97.50%	97.50%	123.66 MB	50.73s
Observations: FedAvg steadily improved across rounds and reached 97.50% accuracy, close to the centralized EfficientNet best result of 98.00%.