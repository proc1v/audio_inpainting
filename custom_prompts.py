few_shot_prompt_template = """
Extract a list objects from user input which needs to be removed from the image.

Example 1
{{"input": "Inpaint the person from the image", "output": ["person"]}}
Example 2
{{"input": "Remove the red car from the image", "output": ["red car"]}}
Example 3
{{"input": "Erase the person walking their dog on the sidewalk.", "output": ["person", "dog"]}}
Example 4
{{"input": "Inpaint the power tower.", "output": ["power tower"]}}
Example 5
{{"input": "Remove the tree and the car from the image", "output": ["tree", "car"]}}

{input}
"""