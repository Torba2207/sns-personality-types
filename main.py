import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
import personsMBTI


def prepare_dataset(data):
    X = []  # Feature matrix
    y = []  # Labels

    for mbti, answer_sets in data.items():
        for answers in answer_sets:
            X.append([1 if ans == "yes" else 0 for ans in answers])  # Convert 'yes'/'no' to 1/0
            y.append(mbti_types.index(mbti))  # Use the index of the MBTI type as the label

    return np.array(X), np.array(y)


def dynamic_prediction(decision_tree, questions):
    print("Welcome! Please answer the following questions with 'yes' or 'no'.")
    current_node = 0  # Start at the root node
    user_answers = []

    while True:
        # Check if the current node is a leaf
        if decision_tree.tree_.children_left[current_node] == -1 and decision_tree.tree_.children_right[
            current_node] == -1:
            predicted_type = mbti_types[np.argmax(decision_tree.tree_.value[current_node])]
            print(f"\nYour predicted personality type is: {predicted_type}")
            break

        # Get the question index for the current node
        feature_index = decision_tree.tree_.feature[current_node]
        question = questions[feature_index]

        # Ask the question and get the user's answer
        answer = input(f"{question} (yes/no): ").strip().lower()
        if answer not in ["yes", "no"]:
            print("Please answer with 'yes' or 'no'.")
            continue

        # Convert the answer to 1/0
        user_answer = 1 if answer == "yes" else 0
        user_answers.append(user_answer)

        # Traverse the tree based on the user's answer
        if user_answer == 0:
            current_node = decision_tree.tree_.children_right[current_node]
        else:
            current_node = decision_tree.tree_.children_left[current_node]


if __name__ == "__main__":
    # Define questions and MBTI types
    questions = [
        "Do you prefer working alone?",
        "Do you rely on intuition more than facts?",
        "Are you emotionally driven?",
        "Do you like planning ahead?",
        "Do you feel drained after socializing for extended periods?",
        "Do you enjoy attending large parties or gatherings?",
        "Do you prefer to reflect on your thoughts before sharing them with others?",
        "Do you often seek solitude to recharge your energy?",
        "Do you enjoy spending time with a few close friends rather than large groups?",
        "Do you prefer to focus on facts and details rather than ideas and concepts?",
        "Do you trust your gut feeling when making decisions?",
        "Do you prefer learning through hands-on experience rather than theoretical knowledge?",
        "Do you find yourself daydreaming or thinking about abstract ideas?",
        "Do you value practical solutions over theoretical possibilities?",
        "Do you prioritize logic and objectivity when making decisions?",
        "Do you value harmony and avoid conflict in relationships?",
        "Do you tend to focus on the bigger picture rather than the details?",
        "Do you make decisions based on how others will feel?",
        "Do you often get emotional or empathetic when hearing about others' problems?",
        "Do you prefer to have a structured, organized environment?",
        "Do you feel comfortable making decisions quickly, even with limited information?",
        "Do you prefer to plan ahead rather than keeping options open?",
        "Do you feel stressed when your plans are disrupted?",
        "Do you often procrastinate and leave tasks until the last minute?",
        "Do you find it easy to adapt to new situations?",
        "Do you prefer to stick to a routine rather than try new things?",
        "Are you more interested in the present moment than planning for the future?",
        "Do you often take the time to analyze your feelings?",
        "Do you enjoy helping others in emotional or personal ways?",
        "Do you often overthink decisions, looking at every possible outcome?",
        "Do you make decisions based on what feels right rather than what seems logical?",
        "Do you prefer having clear guidelines and rules when making decisions?",
        "Do you sometimes regret your decisions after they’ve been made?",
        "Do you prefer working on a project independently rather than as part of a team?",
        "Do you like having clear deadlines and expectations at work?",
        "Do you find it easy to work in an unstructured or dynamic environment?",
        "Do you focus more on the process or the end result of your work?",
        "Do you feel uncomfortable in unfamiliar social situations?",
        "Do you find small talk difficult or unappealing?",
        "Are you the one who often initiates conversations in social settings?",
    ]

    mbti_types = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]

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

    # Train the decision tree
    decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    decision_tree.fit(X, y)

    # Visualize the decision tree with questions as feature names
    plt.figure(figsize=(20, 10))
    plot_tree(
        decision_tree,
        feature_names=questions[:X.shape[1]],  # Use the actual questions as feature names
        class_names=mbti_types,  # MBTI types as class names
        filled=True,
        rounded=True
    )
    plt.show()

    # Start dynamic prediction
    dynamic_prediction(decision_tree, questions)
