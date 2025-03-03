# GenAI for Software Development (Ngram)

* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install Packages](#22-install-packages)  
  * [2.3 Run N-gram](#23-run-n-gram)  
* [3 Report](#3-report) 
* [4 Extra Notes](#4-extra-notes)

---

# **1. Introduction**  
This project explores **code completion in Java**, leveraging **N-gram language modeling**. The N-gram model predicts the next token in a sequence by learning the probability distributions of token occurrences in training data. The model selects the most probable token based on learned patterns, making it a fundamental technique in natural language processing and software engineering automation.  

---

# **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/rachelzheng03/NGramModel.git
```

(2) Navigate into the repository:
```
~ $ cd NGramModel
~/NGramModel $
```

(3) Set up a virtual environment and activate it:

For macOS/Linux:
```
~/NGramModel $ python -m venv ./venv/
~/NGramModel $ source venv/bin/activate
(venv) ~/NGramModel $ 
```

For Windows:
```
python -m venv venv
venv/Scripts/Activate
```

To deactivate the virtual environment, use the command:
```
(venv) $ deactivate
```


## **2.2 Install Packages**

Install the required dependencies:
```
(venv) ~/NGramModel $ pip install -r requirements.txt
```

## **2.3 Run N-gram**

The script has two modes: train and pretrained

(1) Train Mode

The script takes a corpus of Java methods as input and automatically identifies the best-performing model based on a specific N-value. It then evaluates the selected model on the test set extracted according to the assignment specifications.
Since the training corpus differs from both the instructor-provided dataset and our own dataset, we store the results in a file named `results_provided_model.json` to distinguish them accordingly.

Put `<training_filname>.txt` in the folder `data` and run the following command (replace "training_filename" with the name of the file that contains the training corpus):
```
(venv) ~/NGramModel $ python main.py --train <training_filname>.txt
```

The script has an optional command line argument that saves the data of the best-performing model to a JSON file. This file will be saved to the path `./data/saved_models/`. For example, if you want to save the model to a file named `model_data.json` you would run the following command:
```
(venv) ~/NGramModel $ python main.py --train <training_filname>.txt -s model_data.json
```
(2) Pretrained Mode

The pretrained mode skips selecting the best model and goes straight to the testing phase given that a JSON file containing the data of an already trained NGram model is provided. The JSON file must follow the format of `student_model_data.json` which can be found at `./data/saved_models/`. For demonstrating purposes, let the JSON file containing the data of the pretrained model be named `pretrained_model_data.json`. In order to run the script in this mode, put `pretrained_model_data.json` into `./data/saved_models/` and run the following in your terminal:

```
(venv) ~/NGramModel $ python main.py --pretrain pretrained_model_data.json
```

Note that either --train or --pretrain must be specified, and if --pretrain is specified, then -s cannot be used.

## 3. Report
The assignment report is available in the file Assignment_Report.pdf.

## 4. Extra Notes
The JSON file containing the testing results is structured such that the keys indicate the method number, and the values are lists of "tuples" that contain a predicted token and its probability. The first n tokens along with the predicted tokens in each tuple in the list make up the entire predicted function. The generating for the method stops under 3 cases:
1. The NGram model has predicted the end token `<\s>`.
2. The NGram model encounters an ngram it has never seen. The last predicted token for the method in this case is `<UNK>`.
3. The NGram has predicted i tokens such that n+i matches the length of the ground truth method.

The "tuples" are technically strings, since JSON does not support tuples. However, the method `read_test_results` in `util.py` will read the results file and return a dictionary of the testing results where the "tuples" have been converted into actual tuple objects.


