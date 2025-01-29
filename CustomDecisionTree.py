import numpy as np
import matplotlib.pyplot as plt

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, prediction=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold          # Threshold for splitting
        self.left = left                    # Left subtree
        self.right = right                  # Right subtree
        self.prediction = prediction        # Prediction if this is a leaf node


class CustomDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # Check stopping conditions
        if len(set(y)) == 1 or depth >= self.max_depth:
            # Create a leaf node with the majority class
            prediction = np.bincount(y).argmax()
            return DecisionNode(prediction=prediction)

        # Find the best feature and threshold to split on
        best_feature, best_threshold = self._find_best_split(X, y)
        if best_feature is None:
            prediction = np.bincount(y).argmax()
            return DecisionNode(prediction=prediction)

        # Split the dataset
        left_indices = X[:, best_feature] == 1
        right_indices = ~left_indices

        # Recursively build the left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionNode(feature_index=best_feature, threshold=best_threshold,
                            left=left_subtree, right=right_subtree)

    def _find_best_split(self, X, y):
        best_feature = None
        best_gain = 0
        n_features = X.shape[1]

        for feature_index in range(n_features):
            left_mask = X[:, feature_index] == 1
            right_mask = ~left_mask

            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue

            left_labels = y[left_mask]
            right_labels = y[right_mask]

            # Calculate information gain
            gain = self._information_gain(y, left_labels, right_labels)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index

        return best_feature, 0.5  # Threshold is always 0.5 for binary features

    def _information_gain(self, parent, left, right):
        parent_entropy = self._entropy(parent)
        left_entropy = self._entropy(left)
        right_entropy = self._entropy(right)

        left_weight = len(left) / len(parent)
        right_weight = len(right) / len(parent)

        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return -np.sum(probabilities * np.log2(probabilities))

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.prediction is not None:
            return node.prediction
        if x[node.feature_index] == 1:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def display(self, node=None, depth=0, mbti_types=None):
        if mbti_types is None:
            mbti_types = []
        if node is None:
            node = self.root

        if node.prediction is not None:
            print(f"{'|   ' * depth}Predict: {mbti_types[node.prediction]}")
        else:
            print(f"{'|   ' * depth}Feature {node.feature_index}: yes/no?")
            self.display(node.left, depth + 1, mbti_types)
            self.display(node.right, depth + 1, mbti_types)

    def plot_tree(self, questions, mbti_types=None):
        """
        Visualize the decision tree using Matplotlib.

        Parameters:
            questions (list): List of feature questions.
        """
        def _plot(node, x, y, width, depth, ax, mbti_types=mbti_types):
            """
            Recursive helper function to plot the tree.
            """
            if node is None:
                return

            if node.prediction is not None:
                # Leaf node
                label = f"Predict: {mbti_types[node.prediction]}"
                ax.text(x, y, label, ha="center", va="center",
                        bbox=dict(facecolor="lightgreen", edgecolor="black", boxstyle="round"))
            else:
                # Internal node
                label = f"Q: {questions[node.feature_index]}"
                ax.text(x, y, label, ha="center", va="center",
                        bbox=dict(facecolor="lightblue", edgecolor="black", boxstyle="round"))

                # Plot left and right children
                child_y = y - 1
                left_x = x - width / 2
                right_x = x + width / 2

                if node.left:
                    ax.plot([x, left_x], [y, child_y], "k-")
                    _plot(node.left, left_x, child_y, width / 2, depth + 1, ax)

                if node.right:
                    ax.plot([x, right_x], [y, child_y], "k-")
                    _plot(node.right, right_x, child_y, width / 2, depth + 1, ax)

        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        _plot(self.root, x=0, y=0, width=8, depth=0, ax=ax)
        plt.show()

