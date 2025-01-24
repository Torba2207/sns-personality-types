import numpy as np
from tkinter import Tk, Label, Button, Entry, StringVar, messagebox, CENTER
from CustomDecisionTree import CustomDecisionTree as cdt
from questionsMBTI import questions, mbti_types
import personsMBTI


def prepare_dataset(data):
    X = []  # Feature matrix
    y = []  # Labels

    for mbti, answer_sets in data.items():
        for answers in answer_sets:
            X.append([1 if ans == "yes" else 0 for ans in answers])  # Convert 'yes'/'no' to 1/0
            y.append(mbti_types.index(mbti))  # Use the index of the MBTI type as the label

    return np.array(X), np.array(y)


class MBTIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MBTI Personality Predictor")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f8ff")  # Light blue background
        self.custom_tree = None
        self.user_answers = []
        self.current_node = None
        self.current_question_index = 0
        self.tree_depth = StringVar()  # To store user-provided tree depth

        self.main_menu()

    def main_menu(self):
        self.clear_window()

        # Title
        Label(
            self.root, text="MBTI Personality Predictor", font=("Helvetica", 16, "bold"),
            bg="#f0f8ff", fg="#333"
        ).pack(pady=20)

        # Buttons
        Button(
            self.root, text="Training Mode", command=self.training_mode, width=20,
            bg="#4caf50", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
        ).pack(pady=10)
        Button(
            self.root, text="User Application Mode", command=self.user_application_mode, width=20,
            bg="#2196f3", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
        ).pack(pady=10)
        Button(
            self.root, text="Exit", command=self.root.quit, width=20,
            bg="#f44336", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
        ).pack(pady=10)

    def training_mode(self):
        self.clear_window()

        # Training mode label
        Label(
            self.root, text="Training Mode", font=("Helvetica", 16, "bold"),
            bg="#f0f8ff", fg="#333"
        ).pack(pady=10)

        # Input for tree depth
        Label(
            self.root, text="Enter Tree Depth:", font=("Helvetica", 12),
            bg="#f0f8ff", fg="#333"
        ).pack(pady=5)
        Entry(self.root, textvariable=self.tree_depth, font=("Helvetica", 12), width=10).pack(pady=5)

        # Train button
        Button(
            self.root, text="Train", command=self.train_tree, width=20,
            bg="#4caf50", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
        ).pack(pady=10)

        # Back to main menu button
        Button(
            self.root, text="Back to Main Menu", command=self.main_menu, width=20,
            bg="#2196f3", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
        ).pack(pady=10)

    def train_tree(self):
        # Get user-specified depth
        try:
            depth = int(self.tree_depth.get())
            if depth <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for tree depth.")
            return

        # Load Train Data
        data = {}
        for mbti in mbti_types:
            data[mbti] = []
            for i in range(1, 4):
                mbti_low = mbti.lower()
                array_name = f"{mbti_low}{i}"
                if hasattr(personsMBTI, array_name):
                    data[mbti].append(getattr(personsMBTI, array_name))

        # Prepare dataset
        X, y = prepare_dataset(data)

        # Train the custom decision tree
        self.custom_tree = cdt(max_depth=depth)
        self.custom_tree.fit(X, y)

        # Display the decision tree plot
        self.clear_window()
        Label(
            self.root, text=f"Decision Tree (Depth: {depth})", font=("Helvetica", 14, "bold"),
            bg="#f0f8ff", fg="#333"
        ).pack(pady=10)

        # Back to main menu button
        Button(
            self.root, text="Back to Main Menu", command=self.main_menu, width=20,
            bg="#2196f3", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
        ).pack(pady=10)

    def user_application_mode(self):
        if self.custom_tree is None:
            messagebox.showerror("Error", "Train the decision tree first in Training Mode!")
            return

        self.clear_window()
        self.user_answers = []
        self.current_node = self.custom_tree.root
        self.display_question()

    def display_question(self):
        self.clear_window()

        if self.current_node.prediction is not None:
            # Display predicted personality type
            predicted_type = mbti_types[self.current_node.prediction]
            Label(
                self.root, text=f"Your predicted MBTI personality type is: {predicted_type}",
                font=("Helvetica", 14, "bold"), bg="#f0f8ff", fg="#333", wraplength=350, justify=CENTER
            ).pack(pady=20)

            Button(
                self.root, text="Restart", command=self.user_application_mode, width=20,
                bg="#4caf50", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
            ).pack(pady=10)
            Button(
                self.root, text="Back to Main Menu", command=self.main_menu, width=20,
                bg="#2196f3", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
            ).pack(pady=10)
            return

        # Display the current question
        question = questions[self.current_node.feature_index]
        Label(
            self.root, text=question, font=("Helvetica", 12), bg="#f0f8ff", fg="#333",
            wraplength=350, justify=CENTER
        ).pack(pady=20)

        # Buttons for "Yes" and "No"
        Button(
            self.root, text="Yes", command=lambda: self.answer_question(1), width=20,
            bg="#4caf50", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
        ).pack(pady=10)
        Button(
            self.root, text="No", command=lambda: self.answer_question(0), width=20,
            bg="#f44336", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
        ).pack(pady=10)

        # Back button to revisit the previous question
        if self.user_answers:
            Button(
                self.root, text="Go Back", command=self.go_back, width=20,
                bg="#2196f3", fg="white", font=("Helvetica", 12), bd=0, highlightthickness=0
            ).pack(pady=10)

    def answer_question(self, answer):
        self.user_answers.append((self.current_node, answer))

        if answer == 1:
            self.current_node = self.current_node.left
        else:
            self.current_node = self.current_node.right

        self.display_question()

    def go_back(self):
        if self.user_answers:
            self.current_node, _ = self.user_answers.pop()
            self.display_question()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = Tk()
    app = MBTIApp(root)
    root.mainloop()