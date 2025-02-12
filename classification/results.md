2025-02-12 05:32:05,829 - INFO - Using device: cpu
/content/drive/MyDrive/UNSAM_alzheimer/classification/classifier.py:223: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  vae.load_state_dict(torch.load(model_path, map_location=device))
2025-02-12 05:32:18,559 - INFO - Loaded trained BetaVAE from /content/drive/MyDrive/UNSAM_alzheimer/best_beta_vae_model.pth
/content/drive/MyDrive/UNSAM_alzheimer/classification/classifier.py:162: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  data = torch.load(fp, map_location='cpu')
2025-02-12 05:32:27,088 - INFO - Train+Val data shape: (153, 250), Train+Val labels shape: (153,)
2025-02-12 05:32:27,088 - INFO - Test data shape: (31, 250), Test labels shape: (31,)
2025-02-12 05:32:27,089 - INFO - Train+Val label distribution: Counter({1: 80, 0: 73})
2025-02-12 05:32:27,089 - INFO - Test label distribution: Counter({0: 16, 1: 15})
Classifier: Logistic Regression
              precision    recall  f1-score   support

          CN       0.93      0.88      0.90        16
          AD       0.88      0.93      0.90        15

    accuracy                           0.90        31
   macro avg       0.90      0.90      0.90        31
weighted avg       0.91      0.90      0.90        31

--------------------------------------------------
Figure(700x600)
Figure(600x500)
Classifier: SVM (Linear Kernel)
              precision    recall  f1-score   support

          CN       0.89      1.00      0.94        16
          AD       1.00      0.87      0.93        15

    accuracy                           0.94        31
   macro avg       0.94      0.93      0.93        31
weighted avg       0.94      0.94      0.94        31

--------------------------------------------------
Figure(700x600)
Figure(600x500)
Classifier: Random Forest
              precision    recall  f1-score   support

          CN       0.94      0.94      0.94        16
          AD       0.93      0.93      0.93        15

    accuracy                           0.94        31
   macro avg       0.94      0.94      0.94        31
weighted avg       0.94      0.94      0.94        31

--------------------------------------------------
Figure(700x600)
Figure(600x500)
Classifier: k-NN
              precision    recall  f1-score   support

          CN       0.57      0.50      0.53        16
          AD       0.53      0.60      0.56        15

    accuracy                           0.55        31
   macro avg       0.55      0.55      0.55        31
weighted avg       0.55      0.55      0.55        31

--------------------------------------------------
Figure(700x600)
Figure(600x500)
Classifier: Gradient Boosting
              precision    recall  f1-score   support

          CN       1.00      1.00      1.00        16
          AD       1.00      1.00      1.00        15

    accuracy                           1.00        31
   macro avg       1.00      1.00      1.00        31
weighted avg       1.00      1.00      1.00        31

--------------------------------------------------
Figure(700x600)
Figure(600x500)
Classifier: Neural Network
              precision    recall  f1-score   support

          CN       0.94      1.00      0.97        16
          AD       1.00      0.93      0.97        15

    accuracy                           0.97        31
   macro avg       0.97      0.97      0.97        31
weighted avg       0.97      0.97      0.97        31

--------------------------------------------------
Figure(700x600)
Figure(600x500)

=== Performance Summary ===
Classifier: Logistic Regression
  Accuracy: 0.903
  Precision: 0.875
  Recall: 0.933
  F1-Score: 0.903
  ROC AUC: 0.988
----------------------------------------
Classifier: SVM (Linear Kernel)
  Accuracy: 0.935
  Precision: 1.000
  Recall: 0.867
  F1-Score: 0.929
  ROC AUC: 0.946
----------------------------------------
Classifier: Random Forest
  Accuracy: 0.935
  Precision: 0.933
  Recall: 0.933
  F1-Score: 0.933
  ROC AUC: 0.996
----------------------------------------
Classifier: k-NN
  Accuracy: 0.548
  Precision: 0.529
  Recall: 0.600
  F1-Score: 0.562
  ROC AUC: 0.606
----------------------------------------
Classifier: Gradient Boosting
  Accuracy: 1.000
  Precision: 1.000
  Recall: 1.000
  F1-Score: 1.000
  ROC AUC: 1.000
----------------------------------------
Classifier: Neural Network
  Accuracy: 0.968
  Precision: 1.000
  Recall: 0.933
  F1-Score: 0.966
  ROC AUC: 1.000
----------------------------------------
2025-02-12 05:32:30,213 - INFO - Performing GridSearchCV on Random Forest...
Best RandomForest params: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 50}
Evaluation of best RF model:
              precision    recall  f1-score   support

          CN       0.94      1.00      0.97        16
          AD       1.00      0.93      0.97        15

    accuracy                           0.97        31
   macro avg       0.97      0.97      0.97        31
weighted avg       0.97      0.97      0.97        31
