"""
Fixed event labels used across the project.

These labels represent the standardized cricket event types that our
real-time commentary generator will use. All event parsing, retrieval,
generation, and evaluation logic should map to this shared label set.
"""

EVENT_TYPES = [
    "dot_ball",
    "single",
    "double",
    "triple",
    "boundary_four",
    "boundary_six",
    "wicket_bowled",
    "wicket_caught",
    "wicket_lbw",
    "run_out",
    "wide",
    "no_ball",
    "bye_or_legbye",
    "other"
]