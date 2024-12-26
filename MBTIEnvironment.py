import numpy as np
import random

class MBTIEnvironment:
    def __init__(self, mbti_type, questions, mbti_types, simulation_dataset, max_steps):
        """
        Initializes the MBTI environment.
        Args:
            mbti_type (str): The true MBTI type of the simulated user (e.g., "INTJ").
            questions (list of str): The pool of questions the agent can ask.
            max_steps (int): Maximum number of steps allowed in one episode.
        """
        self.mbti_type = mbti_type  # True MBTI type (used for simulation, not in user mode)
        self.questions = questions  # List of questions the agent can ask
        self.asked = [False] * len(questions)
        # State vector initialized to -1 (unasked questions) and one more state for action of prediction
        self.state = [-1] * len(questions)  
        self.steps = 0  # Track the number of steps/questions asked
        self.max_steps = max_steps  # Limit the number of questions
        self.done = False  # Flag to indicate if the episode is finished
        self.personalities = mbti_types
        self.simulation_dataset = simulation_dataset

    def change_personality(self, random_pick, personality_id=0):
        if random_pick:
            self.mbti_type = random.choice(self.personalities)
        else:
            self.mbti_type = self.personalities[personality_id]
            
        

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
        if action == len(self.state)-1:
            predicted_pers = self.predict_personality()
            # print("Prediction initiated...")
            # print(f"Actual Type:{self.mbti_type} Predicted Type:{predicted_pers}")
            if predicted_pers == self.mbti_type:
                self.state[action] = 1
                reward = 150.0
            else:
                self.state[action] = 0
                reward = 50.0
            if self.steps < 5:
                reward -= 20.0
            self.done = True
            return self.state, reward, self.done
        question_id = action  # Get the ID of the current question
        # print(self.questions[question_id])
        # Simulated user response: Simple mapping based on MBTI traits (randomized for training)
        response = self.get_response(self.mbti_type,question_id)  # Yes (1) or No (0)
        # Update state with the response
        self.state[question_id] = response
        # Determine if done (either max steps reached or prediction made)
        self.done = self.steps >= self.max_steps #or self.steps>=len(self.questions)
        reward = 10.0 if response == 1 else 5.0
        reward -= reward * self.steps / self.max_steps
        if self.done:
            reward = 5.0
        elif self.asked[question_id]:
            reward = 1.0
        return self.state, reward, self.done
    
    def reset(self):
        """Resets the environment for a new episode."""
        self.state = [-1] * len(self.questions)  # Reset the state to all questions unasked
        self.steps = 0  # Reset the step count
        self.done = False  # Mark the episode as not done
        self.change_personality(True)
        self.asked = [False] * len(self.questions)
        return self.state
    
    def get_response(self, mbti, question_id):
        sim_person = random.choice(self.simulation_dataset[mbti])
        if sim_person[question_id] == "yes":
            return 1
        return 0

    def predict_personality(self):
        scores = []
        for mbti in self.personalities:
            match_score = sum(
                1 for i in range(len(self.questions) - 1)
                if self.state[i] != -1 and self.state[i] == self.get_response(mbti, i)
            )
            scores.append((mbti, match_score))
        predicted_type = max(scores, key=lambda x: x[1])[0]
        return predicted_type

        


