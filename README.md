# Cost efficient use of BERT in sentiment classification

This code and approach was written and tested on a Hungarian media sentiment corpus, providing a novel (or at least not widely utilized) way of harnessing the power of the BERT language model without extensive resources manifesting in GPU or TPU usage and technical knowledge. It uses the Hubert hungarian pre-trained BERT-model, but utilizies classical ML instead of fine-tuning the model for a downstream classification task. Our approach is not yet pretrained, but will be as soon as a proper corpus becomes available - and hopefully won't need training datasets for future users.

The code was benchmarked against a fine-tuned XLM-Roberta on the same corpus, and reached the following topline results (Roberta result in brackets): 8-way sentiment classification weighted F1: 0.65 [0.73], with a range of category-level F1s of 0.4-0.74 [0.51-0.79]; 3-way classification weighted F1: 0.77 [0.82], 0.58-0.82 [0.51-0.87]. The code was run in a Google Colab GPU-supported free notebook.

The input is needed in a tsv file, containing two necessary columns: "text" for the text itself, and "topik" for the numeric category labels. The code provides a json file for the results compiled in a dictionary, and another one for the optimized parameters.

The code and approach is still under refinement. It was prepared to run in a GPU-supported CUDA environment, but another version is provided for easy usage in a free Google Colab GPU-supported notebook. Minor modifications might be needed to adapt to own directory and data structure.

Written by György Márk Kis and Bálint Sass.
