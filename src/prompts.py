"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           YOUR TASK PROMPTS                                   ║
║                                                                               ║
║  CUSTOMIZE THIS FILE to define prompts/instructions for your task.            ║
║  Prompts are selected based on task type and returned to the model.           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import random


# ══════════════════════════════════════════════════════════════════════════════
#  DEFINE YOUR PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPTS = {
    "default": [
        "Show the fill-to-outline transformation being applied to the second shape. The filled shape should become an outline-only version.",
        "Animate the fill-to-outline transformation where the filled shape changes to outline style according to the established pattern. The question mark should smoothly transition to show the correct outline style.",
        "Complete the visual analogy by showing what the second shape becomes when the same fill-to-outline transformation is applied.",
    ],
    
    "fill_to_outline": [
        "Show the filled shape transforming to outline style. The border thickness should match the example.",
        "Complete the analogy by revealing the outline version of the shape.",
        "Animate the fill-to-outline transformation where the shape loses its fill and becomes outline-only.",
    ],
}


def get_prompt(task_type: str = "default") -> str:
    """
    Select a random prompt for the given task type.
    
    Args:
        task_type: Type of task (key in PROMPTS dict)
        
    Returns:
        Random prompt string from the specified type
    """
    prompts = PROMPTS.get(task_type, PROMPTS["default"])
    return random.choice(prompts)


def get_all_prompts(task_type: str = "default") -> list[str]:
    """Get all prompts for a given task type."""
    return PROMPTS.get(task_type, PROMPTS["default"])
