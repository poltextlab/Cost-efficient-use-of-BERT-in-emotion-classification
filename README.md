# A novel cost-efficient use of BERT embeddings in 8-way emotion classification on a Hungarian media corpus

This code and approach was written and tested on a Hungarian media sentiment corpus, providing a novel (or at least not widely utilized) way of harnessing the power of the BERT language model without extensive resources manifesting in GPU or TPU usage and technical knowledge. It uses the Hubert Hungarian pre-trained BERT-model, but utilizes classical ML instead of fine-tuning the model for a downstream classification task. Our approach is not yet pretrained, but will be as soon as a proper corpus becomes available - and hopefully won't need training datasets for future users.

## Method
Instead of fine-tuning a BERT model, we extract contextual embeddings from the hidden layers and use those as classical inputs for ML approaches.

## Results
The approach was benchmarked against a fine-tuned XLM-Roberta and a fine-tuned Hubert on the same corpus, and reached the following topline results (best Roberta result in brackets): 8-way sentiment classification weighted F1: 0.62 [0.73], with a range of category-level F1s of 0.25-0.71 [0.51-0.79]. The code was run in a Google Colab GPU-supported free notebook.


![image](https://user-images.githubusercontent.com/23291101/145797730-2cd0a4bf-f730-4000-9bb0-c4d053a9438b.png)


### Topline results
|                  |     Hubert    |     Roberta Fine Tuned    |      Hubert Fine Tuned     |
|:----------------:|:-------------:|:-------------------------:|:--------------------------:|
|     Global F1    |      0.62     |            0.73           |             0.71           |


### Weighted F1-scores
|                   |     Hubert    |     Roberta Fine Tuned    |      Hubert Fine Tuned     |
|-------------------|:-------------:|:-------------------------:|:--------------------------:|
|     Anger         |      0.61     |            0.74           |             0.69           |
|     Disgust       |      0.61     |            0.75           |             0.72           |
|     Fear          |      0.25     |            0.71           |             0.50           |
|     Happiness     |      0.32     |            0.67           |             0.45           |
|     Neutral       |      0.51     |            0.51           |             0.59           |
|     Sad           |      0.61     |            0.73           |             0.74           |
|     Successful    |      0.71     |            0.79           |             0.77           |
|     Trustful      |      0.66     |            0.74           |             0.74           |

### Precision
|                   |     Hubert    |     Roberta Fine Tuned    |      Hubert Fine Tuned     |
|-------------------|:-------------:|:-------------------------:|:--------------------------:|
|     Anger         |      0.56     |            0.76           |             0.73           |
|     Disgust       |      0.64     |            0.72           |             0.77           |
|     Fear          |      0.21     |            0.72           |             0.46           |
|     Happiness     |      0.23     |            0.75           |             0.67           |
|     Neutral       |      0.52     |            0.50           |             0.57           |
|     Sad           |      0.62     |            0.73           |             0.71           |
|     Successful    |      0.74     |            0.80           |             0.73           |
|     Trustful      |      0.61     |            0.78           |             0.80           |

### Recall
|                   |     Hubert    |     Roberta Fine Tuned    |      Hubert Fine Tuned     |
|-------------------|:-------------:|:-------------------------:|:--------------------------:|
|     Anger         |      0.67     |            0.71           |             0.66           |
|     Disgust       |      0.59     |            0.79           |             0.67           |
|     Fear          |      0.31     |            0.70           |             0.54           |
|     Happiness     |      0.52     |            0.60           |             0.34           |
|     Neutral       |      0.50     |            0.53           |             0.60           |
|     Sad           |      0.59     |            0.72           |             0.78           |
|     Successful    |      0.68     |            0.77           |             0.81           |
|     Trustful      |      0.70     |            0.71           |             0.68           |

## Usage
The input is needed in a tsv file, containing two necessary columns: "text" for the text itself, and "topik" for the numeric category labels. The code provides a JSON file for the results compiled in a dictionary, and another one for the optimized parameters.

*The code and approach are still under refinement.* It was prepared to run in a GPU-supported CUDA environment. A notebook version is provided for easy usage and debugging in a free Google Colab GPU-supported notebook. Minor modifications might be needed to adapt to own directory and data structure.

## Packages required:
pandas, torch, transformers, numpy, json, sklearn, google.drive (optional)

## Parameters best practice
1. Mean pooling is thought to be the most effective option for extracting contextual embeddings from hidden layers, but this is not a definitive conclusion.
2. Even though there are at least 768 variables for the LR model, the default L2-regularization of sklearn seems to properly take care of this. Previously several dimension reduction techniques were applied and experimented with but none helped with the classification.
3. When using grid search for the LR-model, so far a high number of iterations (such as 6-8000), liblinear solver with L2-regularization, and a relatively narrow band of possible tolerance and C-values (at most 10x change between lower and upper limits) were found to be the most effective.
4. Even though k = 3 is the default for the cross-validation in the script, it can be increased to 5. Further than that possibly increases computing requirements tremendously while not providing notable improvements. The CV-loop runs 3 times by default, this can be changed. As values do not seem to vary much, anything above 9 runs seems unnecessary.

Gy??rgy M??rk Kis, Orsolya Ring, Mikl??s Seb??k: A Novel Cost-efficient Use of BERT Embeddings in 8-way Emotion Classification on a Hungarian Media Corpus (Under review)

The research was supported by the Ministry of Innovation and Technology NRDI Office within the framework of the Artificial Intelligence National Laboratory Program

Thank you for the contributions and support from MTA TK MILAB, SZTAKI, and B??lint Sass.
