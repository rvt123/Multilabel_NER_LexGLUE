# HM_Data_Science

### This repository contains code for the Data Science Assessment organised by the HM Land Registry.

I was supposed to perform one of the Natural Language Programming, Computer Vision, and Graphs task, and I have chosen **Natural Language Programming** task for the assessment. 

#### According to my understanding, I have performed the task in two parts: 
#### - Part 1: Classification of the text from the 1000 test samples. 
#### - Part 2: Extraction of the Location and Date entities from the texts in the test sample. 

## Part 1: Text Classification

Dataset: (ecthr_a) The European Court of Human Rights (ECtHR) hears allegations that a state has breached human rights provisions of the European Convention of Human Rights (ECHR). For each case, the dataset provides a list of factual paragraphs (facts) from the case description. Each case is mapped to articles of the ECHR that were violated (if any). There are 10 classes of voilations and some examples where there are no voilations so total of **11 classes**. This is a **multi-label classification** problem.

Upon understanding the [original code](https://github.com/coastalcph/lex-glue) associated with the research paper and the dataset, it was evident that the Hi-Bert (Hierarchical Bert) performs the best for the task. The authors have used the [OneVsRest](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html) approach to accomplish the multi-label classification using [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html), which acted as a baseline for their research. In my attempt to try something different, I have used **[LightGBM](https://lightgbm.readthedocs.io/en/v3.3.2/)** to perform multi-label classification using multiple approaches **[Classifier Chain](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html)**, **OneVsRest Classifier** and **[MultiOutput Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html#sklearn.multioutput.MultiOutputClassifier)**. The code in **[example_server_HM.py](https://github.com/rvt123/HM_Data_Science/blob/main/example_server_HM.py)** firsts loads the dataset from hugging face using the **datasets** library and preprocesses the train, validation and test examples. It then uses the **[Hyperopt](http://hyperopt.github.io/hyperopt/)** library to find the optimal hyperparameters for the LightGBM model used for classification. To speed up the process of hyperparamter tuning, a technique is devised which uses multiple servers simultaneously to optimise the parameters discussed later.



![This is an image2](https://github.com/rvt123/HM_Data_Science/blob/main/Images/Hyper_tuning_scores.jpg)
