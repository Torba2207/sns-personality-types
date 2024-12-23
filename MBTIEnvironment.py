import numpy as np
import random

class MBTIEnvironment:
    def __init__(self, mbti_type, questions, max_steps=10):
        """
        Initializes the MBTI environment.
        Args:
            mbti_type (str): The true MBTI type of the simulated user (e.g., "INTJ").
            questions (list of str): The pool of questions the agent can ask.
            max_steps (int): Maximum number of steps allowed in one episode.
        """
        self.mbti_type = mbti_type  # True MBTI type (used for simulation, not in user mode)
        self.questions = questions  # List of questions the agent can ask
        self.state = [-1] * len(questions)  # State vector initialized to -1 (unasked questions)
        self.steps = 0  # Track the number of steps/questions asked
        self.max_steps = max_steps  # Limit the number of questions
        self.done = False  # Flag to indicate if the episode is finished

    def changePersonality(self, randomPick, persomalityID=0):
        personalities=[
        "INTJ", "INTP", "ENTJ", "ENTP",  # Rational/Analytical types
        "INFJ", "INFP", "ENFJ", "ENFP",  # Idealistic/Compassionate types
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",  # Guardian/Security-oriented types
        "ISTP", "ISFP", "ESTP", "ESFP"   # Artisan/Adventurous types
        ]
        if randomPick:
            self.mbti_type=random.choice(personalities)
        else:
            self.mbti_type=personalities[persomalityID]
            
        

    def step(self, action):
        """
        Executes an action (asks a question) in the environment.
        Args:
            action (int): Index of the question to ask.
        Returns:
            tuple: (next_state, reward, done)
        """
        if self.done or self.steps >= self.max_steps:
            raise ValueError("Episode is done. Reset the environment.")
        self.steps += 1  # Increment the step count
        question_id = action  # Get the ID of the current question
        # Simulated user response: Simple mapping based on MBTI traits (randomized for training)
        response = np.random.choice([0, 1])  # Yes (1) or No (0)
        # Update state with the response
        self.state[question_id] = response
        # Determine if done (either max steps reached or prediction made)
        self.done = self.steps >= self.max_steps #or self.steps>=len(self.questions)
        reward = -0.1  # Small penalty for each step to encourage fewer questions
        return self.state, reward, self.done
    
    def reset(self):
        """Resets the environment for a new episode."""
        self.state = [-1] * len(self.questions)  # Reset the state to all questions unasked
        self.steps = 0  # Reset the step count
        self.done = False  # Mark the episode as not done
        self.changePersonality(True)
        return self.state


