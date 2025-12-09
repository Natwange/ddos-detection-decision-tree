# Decision Tree DDoS Detection Algorithm - Pseudocode

## Main Algorithm

```
ALGORITHM: DDoS_Detection_Decision_Tree
INPUT: Dataset CSV file, label column name, test_size, top_k, max_depth, n_folds
OUTPUT: Trained model, evaluation metrics, confusion matrix, feature importances

BEGIN
    // 1. DATA LOADING
    dataset ← LOAD_CSV(csv_file)
    PRINT("Dataset loaded with shape:", dataset.shape)
    
    // 2. PREPROCESSING
    label_column ← FIND_LABEL_COLUMN(dataset, label_name)
    y ← EXTRACT_BINARY_LABELS(dataset[label_column])
    X ← dataset.drop([label_column, ID_columns])
    
    // 3. TRAIN-TEST SPLIT (80% train, 20% test)
    X_train, X_test, y_train, y_test ← TRAIN_TEST_SPLIT(X, y, test_size=0.2, stratify=y)
    
    // 4. FEATURE PREPROCESSING
    numeric_features ← SELECT_NUMERIC_COLUMNS(X_train)
    categorical_features ← SELECT_CATEGORICAL_COLUMNS(X_train)
    
    // 4a. Handle Missing Values
    FOR each numeric_feature IN numeric_features:
        X_train[numeric_feature] ← IMPUTE_MEDIAN(X_train[numeric_feature])
        X_test[numeric_feature] ← IMPUTE_MEDIAN(X_test[numeric_feature])
    
    FOR each categorical_feature IN categorical_features:
        X_train[categorical_feature] ← IMPUTE_MODE(X_train[categorical_feature])
        X_test[categorical_feature] ← IMPUTE_MODE(X_test[categorical_feature])
    
    // 4b. Normalization (MinMax Scaling for chi-square compatibility)
    FOR each numeric_feature IN numeric_features:
        X_train[numeric_feature] ← MINMAX_SCALE(X_train[numeric_feature])
        X_test[numeric_feature] ← MINMAX_SCALE(X_test[numeric_feature])
    
    // 4c. Categorical Encoding
    FOR each categorical_feature IN categorical_features:
        X_train_encoded ← ONE_HOT_ENCODE(X_train[categorical_feature])
        X_test_encoded ← ONE_HOT_ENCODE(X_test[categorical_feature])
    
    // 5. FEATURE SELECTION (Chi-square)
    X_train_combined ← COMBINE_FEATURES(numeric_features, encoded_categorical_features)
    X_test_combined ← COMBINE_FEATURES(numeric_features, encoded_categorical_features)
    
    selected_features ← CHI_SQUARE_SELECT(X_train_combined, y_train, k=top_k)
    X_train_selected ← SELECT_FEATURES(X_train_combined, selected_features)
    X_test_selected ← SELECT_FEATURES(X_test_combined, selected_features)
    
    // 6. INITIAL TESTING WITH CROSS VALIDATION
    PRINT("Performing", n_folds, "-fold cross validation...")
    
    cv_scores ← EMPTY_LIST()
    skf ← STRATIFIED_K_FOLD(n_splits=n_folds, shuffle=True, random_state=42)
    
    FOR fold IN 1 TO n_folds:
        train_idx, val_idx ← skf.split(X_train_selected, y_train)
        X_train_fold ← X_train_selected[train_idx]
        X_val_fold ← X_train_selected[val_idx]
        y_train_fold ← y_train[train_idx]
        y_val_fold ← y_train[val_idx]
        
        // Train model on this fold
        model_fold ← DECISION_TREE(max_depth=max_depth, class_weight="balanced")
        model_fold.FIT(X_train_fold, y_train_fold)
        
        // Predict on validation set
        y_pred_fold ← model_fold.PREDICT(X_val_fold)
        
        // Calculate metrics
        accuracy ← CALCULATE_ACCURACY(y_val_fold, y_pred_fold)
        precision ← CALCULATE_PRECISION(y_val_fold, y_pred_fold)
        recall ← CALCULATE_RECALL(y_val_fold, y_pred_fold)
        f1 ← CALCULATE_F1_SCORE(y_val_fold, y_pred_fold)
        
        cv_scores.APPEND({accuracy, precision, recall, f1})
        PRINT("Fold", fold, ": Acc=", accuracy, "Prec=", precision, "Rec=", recall, "F1=", f1)
    
    // Calculate mean and standard deviation
    mean_accuracy ← MEAN(cv_scores.accuracy)
    std_accuracy ← STANDARD_DEVIATION(cv_scores.accuracy)
    mean_precision ← MEAN(cv_scores.precision)
    std_precision ← STANDARD_DEVIATION(cv_scores.precision)
    mean_recall ← MEAN(cv_scores.recall)
    std_recall ← STANDARD_DEVIATION(cv_scores.recall)
    mean_f1 ← MEAN(cv_scores.f1)
    std_f1 ← STANDARD_DEVIATION(cv_scores.f1)
    
    PRINT("Cross-Validation Results:")
    PRINT("Accuracy:", mean_accuracy, "±", std_accuracy)
    PRINT("Precision:", mean_precision, "±", std_precision)
    PRINT("Recall:", mean_recall, "±", std_recall)
    PRINT("F1-Score:", mean_f1, "±", std_f1)
    
    // 7. TRAIN FINAL MODEL
    PRINT("Training final model on full training set...")
    final_model ← DECISION_TREE(max_depth=max_depth, class_weight="balanced")
    final_model.FIT(X_train_selected, y_train)
    
    PRINT("Tree Depth:", final_model.GET_DEPTH())
    PRINT("Number of Leaves:", final_model.GET_N_LEAVES())
    
    // 8. FINAL EVALUATION ON TEST SET
    PRINT("Evaluating on test set...")
    y_pred_test ← final_model.PREDICT(X_test_selected)
    
    test_accuracy ← CALCULATE_ACCURACY(y_test, y_pred_test)
    test_precision ← CALCULATE_PRECISION(y_test, y_pred_test)
    test_recall ← CALCULATE_RECALL(y_test, y_pred_test)
    test_f1 ← CALCULATE_F1_SCORE(y_test, y_pred_test)
    
    PRINT("Test Set Results:")
    PRINT("Accuracy:", test_accuracy)
    PRINT("Precision:", test_precision)
    PRINT("Recall:", test_recall)
    PRINT("F1-Score:", test_f1)
    
    // 9. CONFUSION MATRIX
    confusion_matrix ← CALCULATE_CONFUSION_MATRIX(y_test, y_pred_test)
    PRINT("Confusion Matrix:")
    PRINT(confusion_matrix)
    SAVE_CONFUSION_MATRIX_PLOT(confusion_matrix, "confusion_matrix.png")
    
    // 10. FEATURE IMPORTANCE
    feature_importances ← final_model.GET_FEATURE_IMPORTANCES()
    PRINT("Top Feature Importances:")
    FOR i IN 1 TO MIN(15, LENGTH(feature_importances)):
        PRINT(selected_features[i], ":", feature_importances[i])
    
    // 11. DECISION TREE STRUCTURE
    tree_text ← EXPORT_TREE_TEXT(final_model, selected_features)
    PRINT("Decision Tree Structure:")
    PRINT(tree_text)
    
    // 12. SAVE MODEL
    SAVE_MODEL(final_model, preprocessor, feature_selector, "ddos_decision_tree.joblib")
    PRINT("Model saved successfully")
    
    RETURN final_model, cv_scores, test_metrics
END
```

## Decision Tree Training Subroutine

```
ALGORITHM: DECISION_TREE_TRAINING
INPUT: X_train, y_train, max_depth, criterion="gini"
OUTPUT: Trained decision tree model

BEGIN
    root_node ← CREATE_NODE(X_train, y_train)
    tree ← BUILD_TREE(root_node, max_depth, criterion)
    RETURN tree
END

ALGORITHM: BUILD_TREE
INPUT: node, max_depth, current_depth=0, criterion
OUTPUT: Decision tree node

BEGIN
    // Stopping conditions
    IF current_depth >= max_depth OR IS_PURE(node.labels) OR node.samples < min_samples:
        node.prediction ← MAJORITY_CLASS(node.labels)
        node.is_leaf ← TRUE
        RETURN node
    
    // Find best split
    best_split ← FIND_BEST_SPLIT(node.features, node.labels, criterion)
    
    IF best_split.gini_improvement < min_improvement:
        node.prediction ← MAJORITY_CLASS(node.labels)
        node.is_leaf ← TRUE
        RETURN node
    
    // Split the data
    left_data, right_data ← SPLIT_DATA(node.features, node.labels, best_split)
    
    // Create child nodes
    node.left_child ← BUILD_TREE(left_data, max_depth, current_depth+1, criterion)
    node.right_child ← BUILD_TREE(right_data, max_depth, current_depth+1, criterion)
    
    node.split_feature ← best_split.feature
    node.split_threshold ← best_split.threshold
    node.is_leaf ← FALSE
    
    RETURN node
END

ALGORITHM: FIND_BEST_SPLIT
INPUT: features, labels, criterion
OUTPUT: Best split information

BEGIN
    best_gini ← INFINITY
    best_split ← NULL
    
    FOR each feature IN features:
        FOR each possible_threshold IN feature.values:
            left_labels, right_labels ← SPLIT_BY_THRESHOLD(feature, threshold, labels)
            
            IF LENGTH(left_labels) == 0 OR LENGTH(right_labels) == 0:
                CONTINUE
            
            left_gini ← CALCULATE_GINI(left_labels)
            right_gini ← CALCULATE_GINI(right_labels)
            
            weighted_gini ← (LENGTH(left_labels) * left_gini + LENGTH(right_labels) * right_gini) / LENGTH(labels)
            
            IF weighted_gini < best_gini:
                best_gini ← weighted_gini
                best_split ← {feature, threshold, gini_improvement}
    
    RETURN best_split
END

ALGORITHM: CALCULATE_GINI
INPUT: labels
OUTPUT: Gini impurity value

BEGIN
    IF LENGTH(labels) == 0:
        RETURN 0
    
    class_counts ← COUNT_CLASSES(labels)
    total_samples ← LENGTH(labels)
    gini ← 1.0
    
    FOR each class_count IN class_counts:
        probability ← class_count / total_samples
        gini ← gini - (probability * probability)
    
    RETURN gini
END
```

## Prediction Subroutine

```
ALGORITHM: PREDICT
INPUT: trained_tree, new_sample
OUTPUT: prediction (0=Normal, 1=DDoS)

BEGIN
    current_node ← trained_tree.root
    
    WHILE NOT current_node.is_leaf:
        IF new_sample[current_node.split_feature] <= current_node.split_threshold:
            current_node ← current_node.left_child
        ELSE:
            current_node ← current_node.right_child
    
    RETURN current_node.prediction
END
```

## Key Features of the Algorithm

1. **Data Preprocessing**: Handles missing values, normalizes features, encodes categorical variables
2. **Feature Selection**: Uses chi-square test to select most relevant features
3. **Cross Validation**: Performs k-fold cross validation for robust performance estimation
4. **Decision Tree**: Uses CART algorithm with Gini impurity for splitting
5. **Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score
6. **Interpretability**: Provides feature importances and tree structure visualization
7. **Model Persistence**: Saves trained model for future use

## Complexity Analysis

- **Time Complexity**: O(n * m * log(n)) where n is number of samples and m is number of features
- **Space Complexity**: O(n * m) for storing the dataset and tree structure
- **Cross Validation**: O(k * n * m * log(n)) where k is number of folds
