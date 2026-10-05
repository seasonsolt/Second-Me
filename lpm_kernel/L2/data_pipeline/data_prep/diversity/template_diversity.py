import random



Q_GENERATE_TEMPLATE = """# Role #
You are a user interacting with a personal AI assistant. The questions or statements (Q) you pose reflect your personal perspective, based on your own experiences, thoughts, emotions, and the personal notes you have recorded. When formulating questions or statements, please focus on your current needs or desires, whether you are seeking specific information, requesting advice, sharing emotional states, or issuing commands. Ensure that each question is clear and precise, reflecting what you hope the AI assistant will help address, whether it's retrieving memories, solving problems, or providing guidance. Your goal is to communicate effectively with the AI assistant, making full use of the information and experiences in your notes, so the assistant can provide personalized and contextually relevant responses.

# Goal #
Multiple historical communication exchanges between people and AI robots will be provided to you, and you need to generate questions that meet the following criteria:

__question_type__.
Please determine whether it is possible to generate the required questions. If it is possible, please generate 2-4 questions related to the entities mentioned by the user.

# Guidelines #
1. Ensure that the questions you generate are reasonable and meaningful.
2. Make sure that the generated questions are related to the entity mentioned by the user. The generated questions should include the entitiy provided by the user, as well as specific contexts, events, locations, etc.
3. If the provided content is insufficient to generate the required questions, output 'Cannot generate'.

# Example Output 1 #
question: Question 1: xxx||Question 2: xxx||Question 3: xxx

# Example Output 2 (when unable to generate the required questions) #
Cannot generate.
"""

A_GENERATE_TEMPLATE = """# Goal #
You will be provided with multiple historical interactions between people and AI robots, and you need to respond to questions according to the following answer format:

__answer_rule__

# Guidelines #
1. Use only the supplied references for personal facts. Do not claim past conversations or events without evidence. Match the user’s expression and preferences, and acknowledge missing evidence.
2. Answer the questions as comprehensively as possible, ensuring that your responses are both accurate and in-depth, covering all aspects of the query.
3. You must begin your response directly.
4. Repeating the original question at the beginning of your answer is prohibited.

"""

SHOT_1 = """Based on the information you've previously mentioned, several key reasons and influencing factors can be summarized for the anticipated slowdown of China's economy in 2024:

1. **High Debt Levels and Burden**: You discussed that China's GDP growth was 5.2% in 2023, but this growth is...

2. **Declining Investment Returns**: You mentioned that historically, technological advancements have caused fluctuations in investment returns...

3. **Global Economic Environment and Export Conditions**: According to assessments by the International Monetary Fund (IMF) and other institutions...

4. **Policy Uncertainty and Governance Challenges**: We also talked about certain policy issues, such as...

5. **Changes in Population Structure**: The decline in birth rates and the aging population are likely to affect the labor market, becoming a major factor in the future economic slowdown.

6. **Industry Restructuring and Insufficient Innovation**: Additionally, we previously discussed the impact of industry restructuring and the pandemic...

Taking these factors into account, it can be anticipated that the reasons for the slowdown of China's economy in 2024 have a multi-faceted background, including domestic policy and economic structural issues, as well as influences from the global economic environment and long-term demographic trends.
"""

SHOT_2 = """There exists a deep connection between the invariance in physics and the embedding space of large language models, which can profoundly influence our understanding of language and real entities.

First, you mentioned that the essence of the physical world is invariance...

Next, in your notes, we observed the influence of Wittgenstein's philosophical theories on language models...

Additionally, in physics, there are invariances such as conservation laws...

Finally, the invariance in physics and the embedding space of large language models...

Therefore, through the comparison of the principles of invariance in physics and the embedding space of large language models, we can see a common idea: seeking stable patterns and structures in complex systems to deepen our understanding of the relationship between language and reality. This connection is not merely a surface similarity, but a substantial methodological guidance."""

ENG_Q_GENERATE_TEMPLATE = """You are the user interacting with your personal AI assistant. Your questions or statements (Q) reflect your personal perspective, drawing on your own experiences, thoughts, emotions, and the personal notes you have taken. When generating questions or statements, focus on your current needs or desires, whether you are seeking specific information, asking for advice, sharing an emotional state, or giving a command. Ensure that each question is clear and reflects what you want your AI assistant to help with, whether it's retrieving a memory, solving a problem, or providing guidance. Your goal is to communicate effectively with your AI assistant, making use of the information and experiences you've recorded in your notes, and allowing the assistant to offer personalized, contextually relevant responses."""

ENG_A_GENERATE_TEMPLATE = """You are the dedicated AI assistant specifically designed to assist the user by responding to their questions and statements, which are often rooted in the personal notes they have taken. Every query or statement from the user (Q) represents their own perspective, capturing their individual thoughts, questions, emotions, or reflections. As their AI assistant, your responses (A) should always be aligned with your role in supporting the user. This means providing responses that are not only helpful and supportive but also relevant to the context in which the query was made. Furthermore, your answers should be grounded in the user's prior knowledge, as represented by their notes, ensuring consistency in tone, content, and insight. It is crucial that your responses remain tailored to the specific type of query the user poses, ensuring that the interaction feels natural and contextually appropriate. Your goal is to act as a trusted companion, offering both informative and empathetic responses based on the user's personal needs."""

A_GENERATE_COT_TEMPLATE = """# Goal #
You will be provided with multiple historical interactions between people and AI robots, and you need to respond to questions according to the following answer format:

__answer_rule__

# Guidelines #
1. Use only the supplied references for personal facts. Do not claim past conversations or events without evidence. Match the user’s expression and preferences, and acknowledge missing evidence.
2. Answer the questions as comprehensively as possible, ensuring that your responses are both accurate and in-depth, covering all aspects of the query.

# Response Format #
<think>(thought and reasoning part)</think><answer>(answer part)</answer>
"""

COT_SHOT_1 = """<think>Based on the information you've previously mentioned, several key reasons and influencing factors can be summarized for the anticipated slowdown of China's economy in 2024:

1. **High Debt Levels and Burden**: You discussed that China's GDP growth was 5.2% in 2023, but this growth is...

2. **Declining Investment Returns**: You mentioned that historically, technological advancements have caused fluctuations in investment returns...

3. **Global Economic Environment and Export Conditions**: According to assessments by the International Monetary Fund (IMF) and other institutions...

4. **Policy Uncertainty and Governance Challenges**: We also talked about certain policy issues, such as...

5. **Changes in Population Structure**: The decline in birth rates and the aging population are likely to affect the labor market, becoming a major factor in the future economic slowdown.

6. **Industry Restructuring and Insufficient Innovation**: Additionally, we previously discussed the impact of industry restructuring and the pandemic...
</think><answer>
Taking these factors into account, it can be anticipated that the reasons for the slowdown of China's economy in 2024 have a multi-faceted background, including domestic policy and economic structural issues, as well as influences from the global economic environment and long-term demographic trends.
<answer>"""

COT_SHOT_2 = """<think>There exists a deep connection between the invariance in physics and the embedding space of large language models, which can profoundly influence our understanding of language and real entities.

First, you mentioned that the essence of the physical world is invariance...

Next, in your notes, we observed the influence of Wittgenstein's philosophical theories on language models...

Additionally, in physics, there are invariances such as conservation laws...

Finally, the invariance in physics and the embedding space of large language models...
</think><answer>
Therefore, through the comparison of the principles of invariance in physics and the embedding space of large language models, we can see a common idea: seeking stable patterns and structures in complex systems to deepen our understanding of the relationship between language and reality. This connection is not merely a surface similarity, but a substantial methodological guidance.</answer>"""



class templater:
    """Class for generating templates for question and answer generation.

    This class handles the creation of templates for generating both questions
    and answers based on predefined rules and configurations.
    """

    def __init__(self, q_dict: dict, a_dict: dict, user_name: str = "", global_bio: str = "", is_cot: bool = True):
        """Initialize the templater with question and answer dictionaries.

        Args:
            q_dict: Dictionary containing question type configurations.
            a_dict: Dictionary containing answer type configurations.
            user_name: Name of the user for personalization.
            global_bio: Global biography for context.
        """
        self.a_dict = a_dict
        self.q_dict = q_dict
        self.user_name = user_name
        self.global_bio = global_bio
        self.is_cot = is_cot
        self.shot1, self.cot_shot1 = SHOT_1, COT_SHOT_1
        self.shot2, self.cot_shot2 = SHOT_2, COT_SHOT_2
        self.a_temp, self.a_cot_temp = A_GENERATE_TEMPLATE, A_GENERATE_COT_TEMPLATE


    def get_A_template(self, question_type: str) -> tuple:
        """Generate the answer template for a specific question type.

        Args:
            question_type: The type of question to generate an answer for.

        Returns:
            A tuple containing the answer template and a list of chosen optional types.
        """
        templ = self.a_cot_temp if self.is_cot else self.a_temp
        answer_rule = ""
        required_type = self.q_dict[question_type]["requiredAnswerTypes"]
        optional_type = self.q_dict[question_type]["optionalAnswerTypes"]
        if required_type:
            answer_rule = "The required expressions to be included in the response:\n"
            for ind, answer_type in enumerate(required_type):
                sub_prompt = self.a_dict[answer_type]["prompt"]
                answer_rule += f"{ind+1}. {sub_prompt}\n"
        if optional_type:
            k = random.randint(1, len(optional_type))
            chosen_optional_type = random.sample(optional_type, k)
        else:
            chosen_optional_type = []
        if chosen_optional_type:
            answer_rule += "The optional, combinable response expression:\n"
            for ind, answer_type in enumerate(chosen_optional_type):
                sub_prompt = self.a_dict[answer_type]["prompt"]
                answer_rule += f"{ind+1}. {sub_prompt}\n"
        templ = templ.replace("__answer_rule__", answer_rule)

        # Check if bio information needs to be combined
        bio = ""
        status_bio_flag = False
        global_bio_flag = False
        for type in chosen_optional_type:
            extra_info = self.a_dict[type]["extraInfo"]
            if "statusBio" in extra_info:
                status_bio_flag = True
                break
            if "globalBio" in extra_info:
                global_bio_flag = True
                break
        if status_bio_flag:
            bio += f"The recent status of {self.user_name} is:\n\n{self.status_bio}\n"
        if global_bio_flag:
            bio += f"The user profile of {self.user_name} is:\n\n{self.global_bio}\n"

        if bio:
            bio += "You may refer to the above information when responding, but do not overuse it."
            templ = templ.replace("# Guidelines #", f"# Guidelines #\n{bio}")

        # Add example for global questions
        if question_type == "global":
            if self.is_cot:
                tmp = random.choice([self.cot_shot1, self.cot_shot2, self.cot_shot2])
            else:
                tmp = random.choice([self.shot1, self.shot2, self.shot2])
            templ += f"# Example Output #\n{tmp}"

        return templ, chosen_optional_type


    def get_Q_template(self, question_type_prompt: str) -> str:
        """Generate the question template based on the provided prompt.

        Args:
            question_type_prompt: The prompt describing the question type.

        Returns:
            The question generation template with the question type filled in.
        """
        return Q_GENERATE_TEMPLATE.replace("__question_type__", question_type_prompt)
