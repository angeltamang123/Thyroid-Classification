# Thyroid-Classification

[[Live Demo]](https://thyroid-classification-c6h4nnmey49ptnbtebtyhs.streamlit.app/)

- Missing values in Dataset was addressed using feature dropping in severe missing case and median imputation.
- Numerous prototypes were evaluated on Features scaled on techniques: Normalization, Standardization, Log scaling and Robust Scaling.
- Simple Logistic Regression Model was fitted on Train split with L1 penalty term to shrink coefficients of unimportant terms to zero, this resulted in effective feature selection dropping 29 features to 11 important ones like T3, T4, TSH....
- Models experimented on were: Decision Tree, SVM, MLP, ensembled models like AdaBoost, Gradient Boosting, HistGradient Boosting, Random Forest and Soft Voting Classifer.
- The models were evaluated on Accuracy, Precision, Recall, F1-score, loss-curves, ROC and Precision-Recall curves with emphasis on recall due to the nature of the project as missing positive instances is detrimental.
- Hyper-parameters searched using Grid search from Scikit-learn for Decision Tree for 5-folds. While Cost complexity pruning was utilized for tree based models for regularization purposes and the best CCP alpha value was obtained after 5-fold cross validation.
- Unbalances in classes was address by prototyping across three oversampling techniques: Random, SMOTE and ADASYN
- The finalized model is the Soft voting classifier with smote oversampled Decision Tree, Random Forest and Multi-Layered Perceptron(MLP), achieving recall score of 96.734%.
- The model is saved as a pickle file and utilized in a streamlit app.
