# Arabic Handwriting ANN

## Getting started

1. Use python v3.11.10. Better to create a separate env, if using conda: `conda create -n new_ann python=3.11.10`
2. Install deps: `pip install -r requirements.tsx`
3. Prepare dataset by extracting dataset.zip. In mac / unix bases system can use: `unzip dataset.zip`
4. Tryout running ANN on example_xor.py, which is a sanity check that validates the ANN is learning: `python demo_xor.py`
5. Tryout running ANN on english MNIST dataset: `python demo_mnist.py`
6. Tryout running ANN on our dataset: `python main.py`

## Initial Results

### epoch=1; learning_rate=0.001 (expected to be bad)

```
Dataset ready
Preprocessing...
Setting up neural network...
Training...
Epoch 1/1 - Average Error: 0.114910
Testing...
Sample predictions vs actuals:
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 7, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 2, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 3, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 3, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 6, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 2, Actual: 0
Prediction: 2, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 7, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 6, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Test Accuracy: 10.30%
```

### epoch=10; learning_rate=0.001 (expected to be better than epoch=1)

```Dataset ready
Preprocessing...
Setting up neural network...
Training...
Epoch 1/10 - Average Error: 0.325502
Epoch 2/10 - Average Error: 0.273300
Epoch 3/10 - Average Error: 0.270657
Epoch 4/10 - Average Error: 0.269105
Epoch 5/10 - Average Error: 0.267299
Epoch 6/10 - Average Error: 0.157852
Epoch 7/10 - Average Error: 0.062637
Epoch 8/10 - Average Error: 0.061032
Epoch 9/10 - Average Error: 0.059927
Epoch 10/10 - Average Error: 0.058959
Testing...
Sample predictions vs actuals:
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 6, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 6, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Test Accuracy: 18.20%
```

### epoch=100; learning_rate=0.001 (quite respectable result)

```
Dataset ready
Preprocessing...
Setting up neural network...
Training...
Epoch 1/100 - Average Error: 0.143833
Epoch 2/100 - Average Error: 0.068107
Epoch 3/100 - Average Error: 0.064497
Epoch 4/100 - Average Error: 0.062941
Epoch 5/100 - Average Error: 0.061878
Epoch 6/100 - Average Error: 0.060924
Epoch 7/100 - Average Error: 0.060018
Epoch 8/100 - Average Error: 0.059140
Epoch 9/100 - Average Error: 0.058289
Epoch 10/100 - Average Error: 0.057467
Epoch 11/100 - Average Error: 0.056689
Epoch 12/100 - Average Error: 0.055964
Epoch 13/100 - Average Error: 0.055290
Epoch 14/100 - Average Error: 0.054665
Epoch 15/100 - Average Error: 0.054082
Epoch 16/100 - Average Error: 0.053533
Epoch 17/100 - Average Error: 0.053007
Epoch 18/100 - Average Error: 0.052497
Epoch 19/100 - Average Error: 0.051992
Epoch 20/100 - Average Error: 0.051488
Epoch 21/100 - Average Error: 0.050981
Epoch 22/100 - Average Error: 0.050473
Epoch 23/100 - Average Error: 0.049967
Epoch 24/100 - Average Error: 0.049463
Epoch 25/100 - Average Error: 0.048963
Epoch 26/100 - Average Error: 0.048467
Epoch 27/100 - Average Error: 0.047974
Epoch 28/100 - Average Error: 0.047485
Epoch 29/100 - Average Error: 0.047002
Epoch 30/100 - Average Error: 0.046526
Epoch 31/100 - Average Error: 0.046058
Epoch 32/100 - Average Error: 0.045600
Epoch 33/100 - Average Error: 0.045152
Epoch 34/100 - Average Error: 0.044714
Epoch 35/100 - Average Error: 0.044285
Epoch 36/100 - Average Error: 0.043864
Epoch 37/100 - Average Error: 0.043452
Epoch 38/100 - Average Error: 0.043046
Epoch 39/100 - Average Error: 0.042647
Epoch 40/100 - Average Error: 0.042253
Epoch 41/100 - Average Error: 0.041863
Epoch 42/100 - Average Error: 0.041475
Epoch 43/100 - Average Error: 0.041089
Epoch 44/100 - Average Error: 0.040704
Epoch 45/100 - Average Error: 0.040318
Epoch 46/100 - Average Error: 0.039932
Epoch 47/100 - Average Error: 0.039543
Epoch 48/100 - Average Error: 0.039153
Epoch 49/100 - Average Error: 0.038762
Epoch 50/100 - Average Error: 0.038371
Epoch 51/100 - Average Error: 0.037981
Epoch 52/100 - Average Error: 0.037592
Epoch 53/100 - Average Error: 0.037205
Epoch 54/100 - Average Error: 0.036821
Epoch 55/100 - Average Error: 0.036441
Epoch 56/100 - Average Error: 0.036064
Epoch 57/100 - Average Error: 0.035691
Epoch 58/100 - Average Error: 0.035323
Epoch 59/100 - Average Error: 0.034959
Epoch 60/100 - Average Error: 0.034598
Epoch 61/100 - Average Error: 0.034241
Epoch 62/100 - Average Error: 0.033886
Epoch 63/100 - Average Error: 0.033532
Epoch 64/100 - Average Error: 0.033180
Epoch 65/100 - Average Error: 0.032828
Epoch 66/100 - Average Error: 0.032475
Epoch 67/100 - Average Error: 0.032122
Epoch 68/100 - Average Error: 0.031767
Epoch 69/100 - Average Error: 0.031411
Epoch 70/100 - Average Error: 0.031054
Epoch 71/100 - Average Error: 0.030694
Epoch 72/100 - Average Error: 0.030332
Epoch 73/100 - Average Error: 0.029966
Epoch 74/100 - Average Error: 0.029595
Epoch 75/100 - Average Error: 0.029217
Epoch 76/100 - Average Error: 0.028832
Epoch 77/100 - Average Error: 0.028439
Epoch 78/100 - Average Error: 0.028039
Epoch 79/100 - Average Error: 0.027632
Epoch 80/100 - Average Error: 0.027219
Epoch 81/100 - Average Error: 0.026800
Epoch 82/100 - Average Error: 0.026378
Epoch 83/100 - Average Error: 0.025954
Epoch 84/100 - Average Error: 0.025528
Epoch 85/100 - Average Error: 0.025103
Epoch 86/100 - Average Error: 0.024680
Epoch 87/100 - Average Error: 0.024262
Epoch 88/100 - Average Error: 0.023849
Epoch 89/100 - Average Error: 0.023443
Epoch 90/100 - Average Error: 0.023044
Epoch 91/100 - Average Error: 0.022654
Epoch 92/100 - Average Error: 0.022273
Epoch 93/100 - Average Error: 0.021900
Epoch 94/100 - Average Error: 0.021535
Epoch 95/100 - Average Error: 0.021178
Epoch 96/100 - Average Error: 0.020829
Epoch 97/100 - Average Error: 0.020488
Epoch 98/100 - Average Error: 0.020155
Epoch 99/100 - Average Error: 0.019831
Epoch 100/100 - Average Error: 0.019516
Testing...
Sample predictions vs actuals:
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 1, Actual: 0
Prediction: 0, Actual: 0
Prediction: 9, Actual: 0
Prediction: 0, Actual: 0
Prediction: 5, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Prediction: 0, Actual: 0
Test Accuracy: 87.30%
```
