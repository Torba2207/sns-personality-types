import MBTIEnvironment as menv
from PolicyNetwork import train_policy_network
import PolicyNetwork as polnet
import torch
import torch.optim as optim
import personsMBTI
import random


def save_model(policy_net, filepath="policy_network.pth"):
    torch.save(policy_net.state_dict(), filepath)
    print("Model saved!")

def load_model(policy_net, filepath="policy_network.pth"):
    policy_net.load_state_dict(torch.load(filepath))
    policy_net.eval()
    print("Model loaded!")

if __name__ == "__main__":
    # Define questions and environment
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
    "Guess Personality..."
    #"Do you prefer one-on-one interactions over group settings?"
    ]
    # Predefined Personalities
    mbti_types = [
    "INTJ", "INTP", "ENTJ", "ENTP",
    "INFJ", "INFP", "ENFJ", "ENFP",
    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
    "ISTP", "ISFP", "ESTP", "ESFP"
    ]
    
    # Load Train Data
    data=dict()

    for mbti in mbti_types:
        data[mbti] = []
        for i in range(1,4):
            mbti_low = mbti.lower()
            array_name=f"{mbti_low}{i}"
            if hasattr(personsMBTI, array_name):
                data[mbti].append(getattr(personsMBTI,array_name))

        
    
    # Initialize environment and policy network
    env = menv.MBTIEnvironment(mbti_type=random.choice(mbti_types), questions=questions, mbti_types=mbti_types,
                                simulation_dataset=data, max_steps=10)
    policy_net = polnet.PolicyNetwork(state_size=len(questions), action_size=len(questions))
    optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)

    # Option to load model or train it
    load_existing_model = input("Do you want to load an existing model? (yes/no): ").strip().lower()
    if load_existing_model == 'yes':
        load_model(policy_net)
    else:
        print("Training new model...")
    rewards = train_policy_network(env, policy_net, optimizer)

    save_model(policy_net)