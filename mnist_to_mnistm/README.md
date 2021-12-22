# Commands and results
The goal hereby is to provide examples to get you started with the da package. 
The accuracies achieved here can deviate from the accuracies achieved in the respective papers, since we 
use the same base network and hyperparameter settings (except da method specific ones) for all domain adaptation methods.
Hyperparameters are not tuned extensively.

## Source only (no domain adaptation)

python train_da.py

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9919 | 0.6721

## DANN [1]

python train_da.py --da_type=dann

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9855 | 0.7480

python train_da.py --da_type=dann --da_lambda=0.2 --all_embeds

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9893 | 0.7917

## WDGRL [2]

python train_da.py --da_type=wdgrl

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9861 | 0.8543

## Deep-J-Dot [3]

python train_da.py --da_type=jdot

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9888 | 0.7029

python train_da.py --da_type=jdot --batch_size=128

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9888 | 0.7806

## MMD [4]

python train_da.py --da_type=mmd

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9898 | 0.6778

python train_da.py --da_type=mmd --da_lambda=0.3 --all_embeds

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9912 | 0.7099

## CMD [5]

python train_da.py --da_type=cmd --da_lambda=0.01

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9919 | 0.6445

python train_da.py --da_type=cmd --all_embeds --da_lambda=0.005

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9921 | 0.6649


## SWD [6]

python train_da.py --da_type=swd --da_lambda=0.1

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9929 | 0.6449

python train_da.py --da_type=swd --da_lambda=5

student 1/da1

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9917 | 0.6469

## Coral [7]

python train_da.py --da_type=coral

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9929 | 0.6754

python train_da.py --da_type=coral --all_embeds --da_lambda=0.1

MNIST Accuracy | MNISTM Accuracy
--- | --- |
0.9934 | 0.6771



# References
[1] Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M. & Lempitsky, V. (2016) Domain-adversarial training of neural networks. J. Mach. Learn. Res. 17, 1 (January 2016), 2096–2030.

[2] Shen, J., Qu, Y., Zhang, W. & Yu, Y. (2018) Wasserstein distance guided representation learning for domain adaptation. Proceedings of the 32nd AAAIConference on Artificial Intelligence, New Orleans, FL, USA, Feb. 2018, pp. 4058–4065

[3] Damodara, B. B., Kellenberger, B., Flamary, R., Tuia, D. & Courty, N. (2018) DeepJDOT: Deep Joint Distribution Optimal Transport for Unsupervised Domain Adaptation. European Conference on Computer Vision 2018 (ECCV-2018)

[4] Tzeng, E., Hoffman, J., Zahng, N., Saenko, K. & Darell, T. (2014). Deep Domain Confusion: Maximizing for Domain Invariance. arXiv: 1412.3474

[5] Zellinger, W., Grubinger, T., Lughofer, E., Natschläger, T. & Saminger-Platz, S. (2017). Central Moment Discrepancy (CMD) for Domain-Invariant Representation Learning. International Conference on Learning Representations 2017.

[6] Lee, C. Y., Batra, T., Baig, M. H., & Ulbricht, D. (2019). Sliced wasserstein discrepancy for unsupervised domain adaptation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10285-10295)

[7] Sun, B. & Saenko, K. (2016) Deep coral: Correlation alignment for deep domain adaptation. In Proceedings of Computer Vision – ECCV 2016 Workshops, Gang Hua and Hervé Jégou (Eds.). Springer International Publishing, Cham, 443–450.
