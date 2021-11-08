# Cost efficient use of BERT in 8-way sentiment classification on a Hungarian media corpus

This code and approach was written and tested on a Hungarian media sentiment corpus, providing a novel (or at least not widely utilized) way of harnessing the power of the BERT language model without extensive resources manifesting in GPU or TPU usage and technical knowledge. It uses the Hubert hungarian pre-trained BERT-model, but utilizies classical ML instead of fine-tuning the model for a downstream classification task. Our approach is not yet pretrained, but will be as soon as a proper corpus becomes available - and hopefully won't need training datasets for future users.

## Method
Instead of fine tuning a BERT model, we extract contextual embeddings from the hidden layers and use those as classical inputs for ML-approaches.

## Results
The code was benchmarked against a fine-tuned XLM-Roberta on the same corpus, and reached the following topline results (Roberta result in brackets): 8-way sentiment classification weighted F1: 0.65 [0.73], with a range of category-level F1s of 0.35-0.72 [0.51-0.79]; 3-way classification weighted F1: 0.77 [0.82], 0.58-0.82 [0.51-0.87]. The code was run in a Google Colab GPU-supported free notebook.

![image](https://user-images.githubusercontent.com/23291101/140734165-1ef1e008-b3f9-4b6d-ba19-0454ecf8d510.png)

### Topline results
|                  |     Roberta    |     Hilbert    |     Hubert    |     Roberta Fine Tuned    |
|:----------------:|:--------------:|:--------------:|:-------------:|:-------------------------:|
|     Global F1    |       0.44     |       0.6      |      0.65     |            0.73           |
|     F1-range     |       0.38     |       0.36     |      0.37     |            0.28           |

### Weighted F1-scores
|                   |     Roberta    |     Hilbert    |     Hubert    |     Roberta Fine Tuned    |     N in test set (25%)    |
|-------------------|:--------------:|:--------------:|:-------------:|:-------------------------:|:--------------------------:|
|     Anger         |       0.37     |       0.61     |      0.65     |            0.74           |             157            |
|     Disgust       |       0.47     |       0.62     |      0.64     |            0.75           |             511            |
|     Fear          |       0.14     |       0.29     |      0.35     |            0.71           |              70            |
|     Happiness     |       0.35     |       0.47     |      0.53     |            0.67           |             101            |
|     Neutral       |       0.4      |       0.56     |      0.58     |            0.51           |             383            |
|     Sad           |       0.38     |       0.57     |      0.66     |            0.73           |             491            |
|     Successful    |       0.52     |       0.65     |      0.72     |            0.79           |             782            |
|     Trustful      |       0.49     |       0.65     |      0.67     |            0.74           |             241            |

### Precision
|                   |     Roberta    |     Hilbert    |     Hubert    |     Roberta Fine Tuned    |
|-------------------|:--------------:|:--------------:|:-------------:|:-------------------------:|
|     Anger         |      0.33      |      0.55      |      0.59     |            0.76           |
|     Disgust       |      0.48      |      0.62      |      0.64     |            0.72           |
|     Fear          |      0.11      |      0.25      |      0.32     |            0.72           |
|     Happiness     |      0.28      |      0.39      |      0.44     |            0.75           |
|     Neutral       |      0.42      |      0.55      |      0.59     |            0.5            |
|     Sad           |      0.43      |      0.62      |      0.67     |            0.73           |
|     Successful    |      0.57      |       0.7      |      0.77     |            0.8            |
|     Trustful      |      0.44      |      0.62      |      0.66     |            0.78           |

### Recall
|                   |     Roberta    |     Hilbert    |     Hubert    |     Roberta Fine Tuned    |
|-------------------|:--------------:|:--------------:|:-------------:|:-------------------------:|
|     Anger         |      0.43      |      0.69      |      0.72     |            0.71           |
|     Disgust       |      0.47      |      0.62      |      0.64     |            0.79           |
|     Fear          |      0.23      |      0.36      |      0.39     |            0.7            |
|     Happiness     |      0.49      |      0.59      |      0.65     |            0.6            |
|     Neutral       |      0.38      |      0.56      |      0.57     |            0.53           |
|     Sad           |      0.35      |      0.54      |      0.65     |            0.72           |
|     Successful    |      0.48      |      0.61      |      0.69     |            0.77           |
|     Trustful      |      0.55      |      0.68      |      0.69     |            0.71           |

## Usage
The input is needed in a tsv file, containing two necessary columns: "text" for the text itself, and "topik" for the numeric category labels. The code provides a json file for the results compiled in a dictionary, and another one for the optimized parameters.

The code and approach is still under refinement. It was prepared to run in a GPU-supported CUDA environment, but another (classless) version is provided for easy usage and debugging in a free Google Colab GPU-supported notebook. Minor modifications might be needed to adapt to own directory and data structure.

## Packages required:
pandas, torch, transformers, numpy, json, sklearn, google.drive (optional)

## Parameters best practice
1. Mean pooling is thought to be the most effective option for extracting contextual embeddings from hidden layers, but this is not a definitive conclusion.
2. Even though there are at least 768 variables for the LR model, the default L2-regularization of sklearn seems to properly take care of this. Previously several dimension reduction techniques were applied and experimented with but none helped with the classification.
3. When using grid search for the LR-model, so far a high number of iterations (such as 8000), liblinear solver with L2-regularization, and a relatively narrow band of possible tolerance and C-values (at most 10x change between lower and upper limits) were found to be the most effective.
4. Even though k = 3 is the default for the cross validation in the script, it can be increased to 5. Further than that possibly increases computing requirements tremendously while not providing notable improvements. The CV-loop runs for 3 times by default, this can be changed. As values do not seem to vary much, anything above 9 runs seems unnecessary.


Written by György Márk Kis. Thank you for the contributions and support from SZTAKI, MTA TK MILAB and Bálint Sass.
