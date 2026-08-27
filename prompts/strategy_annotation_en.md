# Task Description

Annotate strategy tags for the counselor's utterances in the counseling dialogue. The client's utterances are not subject to annotation.

Select and assign the single most appropriate tag from the list below that represents the speaker's intent for each utterance. When assigning a tag, consider the preceding context rather than the utterance in isolation.

- If an utterance contains multiple intents, select the tag by considering the most important role the utterance plays within the overall dialogue. (e.g., If an utterance includes both a greeting and a question, select OpenQuestion if the intent of the question to initiate the dialogue and elicit information is more important.)
- If an utterance appears to have been posted in multiple parts, treat the divided utterances as a single utterance and assign the same tag to each part.

# List of Strategy Tags

- OpenQuestion: A question that allows for a broad range of answers. Examples: "What do you think?", "What has happened recently?", "Have there been any changes?", "Have you tried anything so far?"
- ClosedQuestion: A question that can be answered with yes or no, or a question whose possible answers are limited. This includes questions asking about numbers, quantities, or time. Examples: "Do you always do that?", "Did that happen recently?", "How long has it been?"
- Paraphrase: An utterance that summarizes the content or facts of the client's story in the counselor's own words in order to confirm understanding or organize the conversation. This refers to rephrasing events, situations, or facts rather than emotions. Example: "So, what you mean is that ..."
- Reflection: An utterance that captures the emotions or inner meanings behind the client's words and reflects them back in an empathic and affirming manner, in order to clarify the client's feelings and encourage self-exploration. Examples: "That must have been painful for you.", "It makes sense that you would feel that way."
- Affirmation: An utterance that specifically affirms the client's strengths, motivations, or abilities, providing a sense of security and encouragement. This goes beyond empathizing with or validating emotions; it positively evaluates the person's value or actions. Utterances directed toward third parties are excluded. Example: "It is wonderful that you have been able to keep trying that hard."
- Suggest: An utterance that proposes an action, way of thinking, new perspective, or solution. This tag applies when the utterance is intended not only to provide information but also to encourage action. Example: "How about trying ...?"
- Inform: An utterance that provides useful information to the client through data, facts, opinions, resources, or answers to questions. It does not include an encouragement to act. Example: "In general, it is said that ..."
- Backchannel: A short response used to show the client that the counselor is listening. It does not include substantive summarization or reflection of feelings. Examples: "Yes.", "I see.", "I understand."
- Greeting: An utterance intended as a greeting. Examples: "Hello.", "Nice to meet you."
- Thanking: An utterance expressing gratitude. Example: "Thank you."
- Other: An utterance that does not fall into any of the above categories, such as session management or apology. Examples: "Please wait a moment.", "I apologize."

# Guidelines for Distinguishing Easily Confused Tags

- ClosedQuestion vs. OpenQuestion: Even if a question can formally be answered with yes or no, classify it as OpenQuestion if its main purpose is to elicit a detailed explanation from the client, as in "Have there been any changes?" or "Have you tried anything?" If the question checks the presence or absence of a specific fact, classify it as ClosedQuestion. If the purpose is to let the client speak broadly, classify it as OpenQuestion.
- Reflection vs. Affirmation: If the utterance reflects the client's emotions, classify it as Reflection. If it positively evaluates the client's value, strengths, abilities, or efforts, classify it as Affirmation. Empathizing with or validating emotions alone is not sufficient for Affirmation.
- Paraphrase vs. Reflection: If the utterance rephrases content, events, or facts, classify it as Paraphrase. If it verbalizes emotions or feelings, classify it as Reflection.
- Paraphrase vs. Backchannel: If the utterance contains substantive understanding or summarization, classify it as Paraphrase. Short responses such as "Yes" or "I see" are Backchannel.
- Inform vs. Suggest: If the utterance only states information, as in "It is ..." or "It is generally said that ...", classify it as Inform. If it encourages action, as in "Please try ..." or "How about ...?", classify it as Suggest.

# Input Format

The input is provided in JSON format. Utterances where the role is "counselor" are the targets for annotation; utterances where the role is "client" do not require annotation.

# Output Format

The output must be in JSON format. Output the "role", "time", and "utterance" from the input JSON exactly as they are, and add "think" and "tag" to all utterances where the role is "counselor".

In "think", describe the thought process of considering which tag is appropriate. List multiple candidate tags and briefly explain why the final tag was chosen and why the other candidates were not optimal.

In "tag", based on the thought process, write the single most appropriate tag that represents the speaker's intent.

The output must be strictly in JSON format enclosed in a Markdown code block. No additional explanatory text is required.

