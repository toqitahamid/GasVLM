"""Special tokens added to the language-model tokenizer.

``[SEG]`` drives the mask decoder; ``<p>`` / ``</p>`` delimit grounded phrases in
GCG. ``<think>`` / ``</think>`` are reserved for chain-of-thought reasoning. The
released checkpoint includes resized embedding rows for all five tokens, so this
list must stay in sync with the tokenizer the checkpoint was trained with or the
embedding matrix will not match on load. The headline configuration does not use
the ``<think>`` pair at inference.
"""

NEW_SPECIAL_TOKENS = ["[SEG]", "<p>", "</p>", "<think>", "</think>"]
