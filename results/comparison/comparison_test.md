# Model comparison — `test` split (34 documents)

| model                                 |   tier |   n |   macro_f1 |   accuracy |   macro_precision |   macro_recall |   correct | real_correct   |
|:--------------------------------------|-------:|----:|-----------:|-----------:|------------------:|---------------:|----------:|:---------------|
| TF-IDF + LinearSVC                    |      1 |  34 |      0.929 |      0.912 |             0.948 |          0.919 |        31 | 7/10           |
| TF-IDF + LinearSVC [masked]           |      1 |  34 |      0.929 |      0.912 |             0.948 |          0.919 |        31 | 7/10           |
| TF-IDF + Logistic Regression          |      1 |  34 |      0.929 |      0.912 |             0.948 |          0.919 |        31 | 7/10           |
| TF-IDF + Logistic Regression [masked] |      1 |  34 |      0.929 |      0.912 |             0.948 |          0.919 |        31 | 7/10           |
| LayoutLMv3 (fine-tuned)               |      2 |  34 |      0.769 |      0.735 |             0.765 |          0.856 |        25 | 1/10           |
| Llama 3.1 8B (few-shot)               |      3 |  34 |      0.957 |      0.941 |             0.979 |          0.950 |        32 | 8/10           |
| Llama 3.1 8B (zero-shot)              |      3 |  34 |      0.919 |      0.882 |             0.929 |          0.950 |        30 | 8/10           |

## Per-class F1

|                                      |   TF-IDF + LinearSVC |   TF-IDF + LinearSVC [masked] |   TF-IDF + Logistic Regression |   TF-IDF + Logistic Regression [masked] |   LayoutLMv3 (fine-tuned) |   Llama 3.1 8B (few-shot) |   Llama 3.1 8B (zero-shot) |
|:-------------------------------------|---------------------:|------------------------------:|-------------------------------:|----------------------------------------:|--------------------------:|--------------------------:|---------------------------:|
| Exploration License                  |                1.000 |                         1.000 |                          1.000 |                                   1.000 |                     0.750 |                     1.000 |                      1.000 |
| Prospecting Permit                   |                1.000 |                         1.000 |                          1.000 |                                   1.000 |                     0.857 |                     1.000 |                      0.600 |
| Mining Lease                         |                1.000 |                         1.000 |                          1.000 |                                   1.000 |                     0.857 |                     1.000 |                      1.000 |
| Small-Scale Mining Permit            |                0.667 |                         0.667 |                          0.667 |                                   0.667 |                     0.727 |                     0.750 |                      1.000 |
| New License Application              |                0.909 |                         0.909 |                          0.909 |                                   0.909 |                     0.462 |                     0.909 |                      0.750 |
| Renewal Application                  |                0.857 |                         0.857 |                          0.857 |                                   0.857 |                     0.750 |                     1.000 |                      1.000 |
| Environmental Impact Assessment      |                1.000 |                         1.000 |                          1.000 |                                   1.000 |                     1.000 |                     1.000 |                      1.000 |
| Environmental Compliance Certificate |                1.000 |                         1.000 |                          1.000 |                                   1.000 |                     0.750 |                     1.000 |                      1.000 |

## McNemar exact test (pairwise)

| model_a                               | model_b                               |   a_only_correct |   b_only_correct |   p_value |
|:--------------------------------------|:--------------------------------------|-----------------:|-----------------:|----------:|
| TF-IDF + LinearSVC                    | TF-IDF + LinearSVC [masked]           |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC                    | TF-IDF + Logistic Regression          |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC                    | TF-IDF + Logistic Regression [masked] |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC                    | LayoutLMv3 (fine-tuned)               |                7 |                1 |     0.070 |
| TF-IDF + LinearSVC                    | Llama 3.1 8B (few-shot)               |                0 |                1 |     1.000 |
| TF-IDF + LinearSVC                    | Llama 3.1 8B (zero-shot)              |                4 |                3 |     1.000 |
| TF-IDF + LinearSVC [masked]           | TF-IDF + Logistic Regression          |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC [masked]           | TF-IDF + Logistic Regression [masked] |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC [masked]           | LayoutLMv3 (fine-tuned)               |                7 |                1 |     0.070 |
| TF-IDF + LinearSVC [masked]           | Llama 3.1 8B (few-shot)               |                0 |                1 |     1.000 |
| TF-IDF + LinearSVC [masked]           | Llama 3.1 8B (zero-shot)              |                4 |                3 |     1.000 |
| TF-IDF + Logistic Regression          | TF-IDF + Logistic Regression [masked] |                0 |                0 |     1.000 |
| TF-IDF + Logistic Regression          | LayoutLMv3 (fine-tuned)               |                7 |                1 |     0.070 |
| TF-IDF + Logistic Regression          | Llama 3.1 8B (few-shot)               |                0 |                1 |     1.000 |
| TF-IDF + Logistic Regression          | Llama 3.1 8B (zero-shot)              |                4 |                3 |     1.000 |
| TF-IDF + Logistic Regression [masked] | LayoutLMv3 (fine-tuned)               |                7 |                1 |     0.070 |
| TF-IDF + Logistic Regression [masked] | Llama 3.1 8B (few-shot)               |                0 |                1 |     1.000 |
| TF-IDF + Logistic Regression [masked] | Llama 3.1 8B (zero-shot)              |                4 |                3 |     1.000 |
| LayoutLMv3 (fine-tuned)               | Llama 3.1 8B (few-shot)               |                1 |                8 |     0.039 |
| LayoutLMv3 (fine-tuned)               | Llama 3.1 8B (zero-shot)              |                2 |                7 |     0.180 |
| Llama 3.1 8B (few-shot)               | Llama 3.1 8B (zero-shot)              |                4 |                2 |     0.688 |
