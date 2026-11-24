# Model comparison — `test` split (34 documents)

| model                                 |   tier |   n |   macro_f1 |   accuracy |   macro_precision |   macro_recall |   correct | real_correct   |
|:--------------------------------------|-------:|----:|-----------:|-----------:|------------------:|---------------:|----------:|:---------------|
| TF-IDF + LinearSVC                    |      1 |  34 |      0.929 |      0.912 |             0.948 |          0.919 |        31 | 7/10           |
| TF-IDF + LinearSVC [masked]           |      1 |  34 |      0.929 |      0.912 |             0.948 |          0.919 |        31 | 7/10           |
| TF-IDF + Logistic Regression          |      1 |  34 |      0.929 |      0.912 |             0.948 |          0.919 |        31 | 7/10           |
| TF-IDF + Logistic Regression [masked] |      1 |  34 |      0.929 |      0.912 |             0.948 |          0.919 |        31 | 7/10           |

## Per-class F1

|                                      |   TF-IDF + LinearSVC |   TF-IDF + LinearSVC [masked] |   TF-IDF + Logistic Regression |   TF-IDF + Logistic Regression [masked] |
|:-------------------------------------|---------------------:|------------------------------:|-------------------------------:|----------------------------------------:|
| Exploration License                  |                1.000 |                         1.000 |                          1.000 |                                   1.000 |
| Prospecting Permit                   |                1.000 |                         1.000 |                          1.000 |                                   1.000 |
| Mining Lease                         |                1.000 |                         1.000 |                          1.000 |                                   1.000 |
| Small-Scale Mining Permit            |                0.667 |                         0.667 |                          0.667 |                                   0.667 |
| New License Application              |                0.909 |                         0.909 |                          0.909 |                                   0.909 |
| Renewal Application                  |                0.857 |                         0.857 |                          0.857 |                                   0.857 |
| Environmental Impact Assessment      |                1.000 |                         1.000 |                          1.000 |                                   1.000 |
| Environmental Compliance Certificate |                1.000 |                         1.000 |                          1.000 |                                   1.000 |

## McNemar exact test (pairwise)

| model_a                      | model_b                               |   a_only_correct |   b_only_correct |   p_value |
|:-----------------------------|:--------------------------------------|-----------------:|-----------------:|----------:|
| TF-IDF + LinearSVC           | TF-IDF + LinearSVC [masked]           |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC           | TF-IDF + Logistic Regression          |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC           | TF-IDF + Logistic Regression [masked] |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC [masked]  | TF-IDF + Logistic Regression          |                0 |                0 |     1.000 |
| TF-IDF + LinearSVC [masked]  | TF-IDF + Logistic Regression [masked] |                0 |                0 |     1.000 |
| TF-IDF + Logistic Regression | TF-IDF + Logistic Regression [masked] |                0 |                0 |     1.000 |
