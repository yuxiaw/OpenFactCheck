"""Prompts for generating hallucinations."""

EVIDENCE_HALLUCINATION = """Generate a paragraph that answers the question.

Question: What is New York-Style pizza?
Text: New York-style pizza has slices that are large and wide with a thin crust that is foldable yet crispy. It is traditionally topped with tomato sauce and mozzarella cheese, with any extra toppings placed on top of the cheese.

Question: When did the first McDonald's open?
Text: The McDonald's brothers opened their first McDonald's restaurant in 1940 in San Bernardino, California. Originally, a carhop drive-in system was used to serve customers. The initial menu items were centered around barbecue and the first name the brothers called their business was "McDonald's Famous Barbecue."

Question: {query}
Text:
""".strip()
