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


## EXP-001