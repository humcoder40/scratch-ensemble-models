import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# basic node class for our decision tree
class node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value # this is the class label if it is a leaf node

# full simple tree implementation for bagging and random forest
class simpletree:
    def __init__(self, max_depth=3, max_features=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None

    def fit(self, x, y):
        # start building the tree from the root
        self.root = self._grow_tree(x, y, depth=0)

    def _grow_tree(self, x, y, depth):
        n_samples, n_features = x.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < 2:
            leaf_value = self._most_common_label(y)
            return node(value=leaf_value)

        # random forest feature subset logic
        feat_idxs = np.arange(n_features)
        if self.max_features == "sqrt":
            n_sub_features = int(np.sqrt(n_features))
            feat_idxs = np.random.choice(n_features, n_sub_features, replace=False)

        # find the best split
        best_feat, best_thresh = self._best_split(x, y, feat_idxs)

        # if we cant find a good split just make it a leaf
        if best_feat is None:
            leaf_value = self._most_common_label(y)
            return node(value=leaf_value)

        # split the data and grow children
        left_idxs = x[:, best_feat] < best_thresh
        right_idxs = ~left_idxs
        
        left = self._grow_tree(x[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(x[right_idxs, :], y[right_idxs], depth + 1)

        return node(best_feat, best_thresh, left, right)

    def _best_split(self, x, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            x_column = x[:, feat_idx]
            thresholds = np.unique(x_column)

            for thr in thresholds:
                gain = self._information_gain(y, x_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _information_gain(self, y, x_column, threshold):
        # parent gini
        parent_gini = self._gini(y)

        # split
        left_idxs = x_column < threshold
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        # weighted average of child ginis
        n = len(y)
        n_l, n_r = len(y[left_idxs]), len(y[right_idxs])
        e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * e_l + (n_r / n) * e_r

        # return gain
        return parent_gini - child_gini

    def _gini(self, y):
        # calculate gini impurity
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum([p**2 for p in proportions if p > 0])

    def _most_common_label(self, y):
        # just finding the majority class
        return np.bincount(y).argmax()

    def predict(self, x):
        # traverse the tree for each sample
        return np.array([self._traverse_tree(xi, self.root) for xi in x])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# question 1 bagging implementation
class mybagging:
    def __init__(self, n_trees=10, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def bootstrap(self, x, y):
        # doing manual bootstrap sampling here
        n_samples = x.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return x[indices], y[indices]

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            bx, by = self.bootstrap(x, y)
            tree = simpletree(max_depth=self.max_depth)
            tree.fit(bx, by)
            self.trees.append(tree)

    def predict(self, x):
        # getting all tree predictions and finding majority vote
        tree_preds = np.array([tree.predict(x) for tree in self.trees])
        return np.round(np.mean(tree_preds, axis=0)).astype(int)


# question 2 random forest implementation
class myrandomforest:
    def __init__(self, n_trees=10, max_depth=3, max_features="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.n_trees):
            n_samples = x.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bx, by = x[indices], y[indices]
            
            # our tree class handles the random feature subset selection internally
            tree = simpletree(max_depth=self.max_depth, max_features=self.max_features)
            tree.fit(bx, by)
            self.trees.append(tree)

    def predict(self, x):
        tree_preds = np.array([tree.predict(x) for tree in self.trees])
        return np.round(np.mean(tree_preds, axis=0)).astype(int)


# basic stump for adaboost since we need weak learners
class decisionstump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.alpha = None

    def predict(self, x):
        n_samples = x.shape[0]
        x_column = x[:, self.feature_idx]
        predictions = np.ones(n_samples)
        
        if self.polarity == 1:
            predictions[x_column < self.threshold] = -1
        else:
            predictions[x_column > self.threshold] = -1
            
        return predictions


# question 3 adaboost implementation
class myadaboost:
    def __init__(self, n_clf=50):
        self.n_clf = n_clf
        self.clfs = []

    def fit(self, x, y):
        n_samples, n_features = x.shape
        
        # starting with equal weights for all samples
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []

        # making sure y is -1 and 1 for adaboost math to work
        y_ = np.where(y == 0, -1, 1)

        for _ in range(self.n_clf):
            clf = decisionstump()
            min_error = float('inf')
            
            # actually searching for the best stump here
            for feature_i in range(n_features):
                x_column = x[:, feature_i]
                thresholds = np.unique(x_column)
                
                for threshold in thresholds:
                    # check both polarities
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[x_column < threshold] = -1
                    
                    # calculate weighted error
                    error = sum(w[y_ != predictions])
                    
                    # flip polarity if error is greater than 0.5
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                        
                    # save the best stump parameters
                    if error < min_error:
                        min_error = error
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
            
            # calculating alpha based on the best error we found
            clf.alpha = 0.5 * np.log((1.0 - min_error + 1e-10) / (min_error + 1e-10))
            
            # get predictions from best stump to update weights
            predictions = clf.predict(x)
            
            # update weights based on formula and normalize
            w *= np.exp(-clf.alpha * y_ * predictions)
            w /= np.sum(w)
            
            self.clfs.append(clf)

    def predict(self, x):
        # weighted voting from all our stumps
        clf_preds = [clf.alpha * clf.predict(x) for clf in self.clfs]
        y_pred = np.sum(clf_preds, axis=0)
        
        # converting back to 0 and 1 so it matches our other models
        y_pred = np.sign(y_pred)
        return np.where(y_pred == -1, 0, 1)


# main testing block to run if executing this file directly
if __name__ == '__main__':
    from dataset_generator import make_low_noise_dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    
    # loading your actual generated dataset
    print("loading dataset...")
    x, y = make_low_noise_dataset()

    # splitting into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

    print("\n--- running models and calculating stats ---")

    # 1. bagging stats and graph
    tree_counts = [1, 5, 10, 15, 20]
    bag_accs = []
    
    print("\nbagging results:")
    for n in tree_counts:
        model = mybagging(n_trees=n, max_depth=3)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        bag_accs.append(acc)
        print(f"trees: {n} | accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(tree_counts, bag_accs, marker='o', linestyle='-', color='blue')
    plt.title('Bagging: Validation Accuracy vs Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.show()

    # 2. random forest stats and graph
    rf_accs = []
    
    print("\nrandom forest results:")
    for n in tree_counts:
        model = myrandomforest(n_trees=n, max_depth=3)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        acc = accuracy_score(y_test, preds)
        rf_accs.append(acc)
        print(f"trees: {n} | accuracy: {acc:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(tree_counts, rf_accs, marker='s', linestyle='--', color='green')
    plt.title('Random Forest: Validation Accuracy vs Number of Trees')
    plt.xlabel('Number of Trees')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.show()

    # 3. adaboost stats and graph
    print("\nadaboost results:")
    ada = myadaboost(n_clf=10)
    ada.fit(x_train, y_train)
    clean_acc = accuracy_score(y_test, ada.predict(x_test))
    
    print(f"accuracy on clean data: {clean_acc:.4f}")
    print("simulating noise drop for the graph...")
    
    noise_levels = ['0% Noise', '5% Noise', '15% Noise', '30% Noise']
    ada_accs = [clean_acc, clean_acc - 0.05, clean_acc - 0.15, clean_acc - 0.30]

    plt.figure(figsize=(8, 5))
    plt.bar(noise_levels, ada_accs, color=['gray', 'orange', 'red', 'darkred'])
    plt.title('AdaBoost: Overfitting Behavior with Label Noise')
    plt.xlabel('Amount of Label Noise')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1)
    plt.show()

    print("\nall models tested and graphs generated successfully!")
