# HM_Data_Science

### This repository contains code for the Data Science Assessment organised by the HM Land Registry.

I was supposed to perform one of the Natural Language Programming, Computer Vision, and Graphs task, and I have chosen **Natural Language Programming** task for the assessment. 

#### According to my understanding, I have performed the task in two parts: 
#### - Part 1: Classification of the text from the 1000 test samples. 
#### - Part 2: Extraction of the Location and Date entities from the texts in the test sample. 

## Part 1: Text Classification

Dataset: (ecthr_a) The European Court of Human Rights (ECtHR) hears allegations that a state has breached human rights provisions of the European Convention of Human Rights (ECHR). For each case, the dataset provides a list of factual paragraphs (facts) from the case description. Each case is mapped to articles of the ECHR that were violated (if any). There are 10 classes of voilations and some examples where there are no voilations so total of **11 classes**. This is a **multi-label classification** problem.

Upon understanding the orginal code associatiated with the research paper and the dataset, it was quite evident that the Hi-Bert (Hierarchial Bert) performs the best for task. The authors have used OneVsRest approach to perform the multi-label classification using LinearSVC. This acted as a baseline for their research. In my attempt to try something different, I have used LightGBM to perform multi-label classfication using multiple approaches Classifier Chain, OneVsRest Classfier and MultiOutput Classifier.
The code in [example_server_HM.py](https://github.com/rvt123/HM_Data_Science/blob/main/example_server_HM.py) firsts load the dataset from hugging face using the datasets library and preprocesses the train, validation and test examples. It then uses the hyperopt library to find the optimal hyperparamers for the LightGBM model used for classification. To speed up the process of hyperparamter tuning, a technique is devised which uses multiple servers at the same time to optimise the paramteres which is discussed later.
