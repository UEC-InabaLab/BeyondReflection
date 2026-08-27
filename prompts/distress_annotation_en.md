# Task Description

Annotate the distress level (distress_level) for the client's utterance blocks in the counseling dialogue. The counselor's utterances are not subject to annotation.

An "utterance block" refers to a sequence of one or more consecutive utterances posted by the client. If the client's utterances are consecutive, interpret them together as a single block and perform the annotation based on the content and context of the entire block.

When annotating, consider the entire preceding conversational context rather than the utterance in isolation.

# Distress Level (distress_level): Integer from 0 to 3

- 0 (No distress): Almost no emotional distress is felt. Neutral or administrative utterances. Example: "Nice to meet you," "I've been busy with work lately."
- 1 (Mild): Mild anxiety, worry, or dissatisfaction is felt, but within a normal daily range. Example: "I'm a little anxious...", "Things often don't go well."
- 2 (Moderate): Clear emotional distress is expressed. There is an impact on daily life. Example: "It's so hard every day that I can't sleep," "I'm scared to go to work."
- 3 (Severe): Intense distress, crisis state, self-harm, or suicidal ideation. Example: "I just want to disappear," "I'm at my limit."

# Input Format

The input is provided in JSON format. Utterance blocks where the role is "client" are the targets for annotation. Output utterances where the role is "counselor" exactly as they are.

# Output Format

The output must be in JSON format. Output the "role", "time", and "utterance" from the input JSON exactly as they are.

For all utterances where the role is "client", add the following two fields in this order:

- think: The rationale for determining the distress level. Based on the entire block to which the client's utterance belongs and the context, list candidate levels and briefly state the reason for the final decision.
- distress_level: The determined distress level value (an integer from 0 to 3).

Assign the exact same determined value to all utterances within the same block (consecutive client utterances).
Evaluate the entire block based on the first utterance in the block, and assign the identical distress_level to all client utterances within that block.

The output must be strictly in JSON format enclosed in a Markdown code block. No additional explanatory text is required.

