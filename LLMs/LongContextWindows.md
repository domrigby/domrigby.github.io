# Evaluating Long Context (Reasoning) Ability

Date read: 18th October 2025

[Blog post]()

## Key Points
* Reasoning performance tends to decrease with context lenght, long before max context length is reached. This is especialy important when doing CoT as this generates a bunch more tokens.
* E.g. ChatGPT-4 decays at >100K tokens
* Measuring this ability tends to come in two parts:
	1. Determine which parts of the text are relevant 
	2. Perform tasks required
* Examples:
	* Needle in a haystack: 
		* Hide a bit of info in a large block of text and get the LLM to retrieve it
		* Checks last token can attend to first.

	* NIAH within distribution:
		* Same as above, but info should be non-obvious part of the text (in distribution)
		* More difficult as info doesn't stand out

	* NIAH with reasoning:
		* Model has to reason with info provided across context window

	* NIAH with reasoning across whole context window

	* Gives their example of LongCodeEdit: a bug fixing task across a large codebase.

* Long story short, they differ on how difficult it is to determine which parts are relevant and then how easy it is to process this relevant information.