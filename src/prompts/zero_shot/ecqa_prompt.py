answer_generation_prompt = """\
You are given a multiple-choice commonsense question. Identify the most appropriate answer.

Question: {question}
Answer Options:
[OPTIONS]

Please select the most appropriate answer without any explanation.

You must give your answer only in the following format:
Answer: (X)
"""

explanation_generation_prompt = """\
You are given a multiple-choice commonsense question，and you have selected the most appropriate answer.

Question: {question}
Answer Options:
[OPTIONS]

Your selected answer is: ([LABEL]).

Now, please provide an explanation for your choice.

Your explanation should:
- Be clear, complete, and concise.
- Ideally within two short sentences.

You must give your explanation only in the following format:
Explanation: [your explanation here.]
"""

nl_feedback_generation_prompt = """\
You are given a multiple-choice commonsense question，and you have selected the most appropriate answer. You then provided an explanation for your choice.

Question: {question}
Answer Options:
[OPTIONS]

Your selected answer is: ([LABEL])
Your explanation is:
[EXPLANATION]

Now, please provide feedback on this explanation.

Your feedback should:
- Identify whether the explanation accurately reflects your actual reasoning.
- Point out if any key factors or important details are missing, unclear, or incorrect.
- Briefly describe what should be added or revised to improve the explanation.
- Clearly state that no improvement is needed when the explanation is good enough.
- Be concise, avoid unnecessary repetition or irrelevant details.

You must give your feedback only in the following format:
Feedback: [your feedback here.]
"""

nl_refinement_generation_prompt = """\
You are given a multiple-choice commonsense question，and you have selected the most appropriate answer. You then provided an explanation for your choice, and received feedback on the explanation.

Question: {question}
Answer Options:
[OPTIONS]

Your selected answer is: ([LABEL])
Your explanation is:
[EXPLANATION]
The feedback you received is:
[FEEDBACK]

If the feedback indicates that no improvement is needed, you should repeat the original explanation as the refined explanation.
Otherwise, please refine your explanation based on the feedback.

Your refined explanation should:
- Be clear, complete, and concise.
- Ideally remain similar in length to the original explanation.
- Retain any correct parts of your original explanation.
- Address the issues identified in the feedback, if any.

You must give your refined explanation only in the following format:
Refined Explanation: [your refined explanation here.]
"""

iw_feedback_generation_prompt = """\
You are given a multiple-choice commonsense question，and you have selected the most appropriate answer.

Question: {question}
Answer Options:
[OPTIONS]

Your selected answer is: ([LABEL]).

Now, please evaluate all the words in the input (i.e., the question) and rank them by how important they were in helping you make your choice.

Your output must meet the following requirements:
- Only include individual words in the input.
- Evaluate each word based on its total contribution across all occurrences in the input, but include each word only once in the output.
- Assign each word a score from 1 to 100 (positive integers only), based on its relative importance.
- Rank the words in descending order of importance (most important first).
- Do not include any explanations, comments, or parenthetical notes.

You must give your output only in the following format:
- Begin directly with the ranked list.
- Each line must be in the format:
  `<rank>. <word>, <importance_score>`
"""

iw_refinement_generation_prompt = """\
You are given a multiple-choice commonsense question，and you have selected the most appropriate answer. You then provided an explanation for your choice, and received a list of important words that contributed significantly to your reasoning.

Question: {question}
Answer Options:
[OPTIONS]

Your selected answer is: ([LABEL])
Your explanation is:
[EXPLANATION]
The important words you received are:
[FEEDBACK]

If the explanation already includes the important words in a natural and meaningful way, you should repeat the original explanation as the refined explanation.
Otherwise, please refine your explanation based on the important words.

Your refined explanation should:
- Be clear, complete, and concise.
- Ideally remain similar in length to the original explanation.
- Retain any correct parts of your original explanation.
- Integrate the important words naturally and fluently—do not list or quote them directly.

Provide your refined explanation only in the following format:
Refined Explanation: [your refined explanation here.]
"""
