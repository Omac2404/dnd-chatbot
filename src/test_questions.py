"""
RAG sistemi için test soruları ve expected answers
"""

TEST_QUESTIONS = [
    {
        "question": "What are the six ability scores?",
        "category": "Basic Rules",
        "expected_keywords": ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]
    },
    {
        "question": "How do I calculate armor class?",
        "category": "Combat",
        "expected_keywords": ["armor", "dexterity", "modifier", "ac"]
    },
    {
        "question": "What is a saving throw?",
        "category": "Mechanics",
        "expected_keywords": ["d20", "ability", "save", "dc"]
    },
    {
        "question": "What races can I play?",
        "category": "Character Creation",
        "expected_keywords": ["dwarf", "elf", "halfling", "human"]
    },
    {
        "question": "How does initiative work in combat?",
        "category": "Combat",
        "expected_keywords": ["dexterity", "roll", "order", "turn"]
    },
    {
        "question": "What is the difference between a spell attack and a saving throw spell?",
        "category": "Spellcasting",
        "expected_keywords": ["spell", "attack", "save", "dc"]
    },
    {
        "question": "How do I gain experience points?",
        "category": "Advancement",
        "expected_keywords": ["xp", "level", "encounter", "milestone"]
    },
    {
        "question": "What is proficiency bonus?",
        "category": "Basic Rules",
        "expected_keywords": ["level", "bonus", "proficient", "add"]
    },
]