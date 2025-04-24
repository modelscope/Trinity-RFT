DEFAULT_CLEANER = {
    "quality": {
        "alphanumeric_filter": "Filter text with alphabet/numeric ratio out of specific range.",
        "character_repetition_filter": "Filter text with the character repetition ratio out of specific range.",
        "flagged_words_filter": "Filter text with the flagged-word ratio larger than a specific max value.",
        "special_characters_filter": "Filter text with special-char ratio out of specific range.",
        "stopwords_filter": "Filter text with stopword ratio smaller than a specific min value.",
        "word_repetition_filter": "Filter text with the word repetition ratio out of specific range.",
        "llm_quality_score_filter": "Filter to keep sample with high quality score estimated by LLM.",
    },
    "difficulty": {
        "perplexity_filter": "Filter text with perplexity score out of specific range.",
        "language_id_score_filter": "Filter text in specific language with language scores larger than a specific max value.",
        "llm_difficulty_score_filter": "Filter to keep sample with high difficulty score estimated by LLM.",
    },
}

DEFAULT_HUMAN_ANNOTATOR = {
    "human_preference_annotation_mapper": "Operator for human preference annotation using Label Studio.",
}

DEFAULT_SYNTHESIZER = {}

# For cleaner
DIMENSION_STATS_KEYS = {
    "quality_score": {
        "alnum_ratio": {"better": "higher", "range": [0.0, 1.0]},
        "char_rep_ratio": {"better": "lower", "range": [0.0, 1.0]},
        "flagged_words_ratio": {"better": "lower", "range": [0.0, 1.0]},
        "special_char_ratio": {"better": "lower", "range": [0.0, 1.0]},
        "stopwords_ratio": {"better": "higher", "range": [0.0, 1.0]},
        "word_rep_ratio": {"better": "lower", "range": [0.0, 1.0]},
        "llm_quality_score": {"better": "higher", "range": [0.0, 1.0]},
    },
    "difficulty_score": {
        "perplexity": {"better": "higher", "range": [0.0, None]},
        "lang_score": {"better": "lower", "range": [0.0, 1.0]},
        "llm_difficulty_score": {"better": "higher", "range": [0.0, 1.0]},
    },
}


# For calling llm_quality_score_filter and llm_difficulty_score_filter following environment key should be set.
# export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1/
# export OPENAI_API_KEY=your_dashscope_key
DEFAULT_OP_ARGS = {
    # cleaners
    "flagged_words_filter": {
        "lang": "en",
    },
    "stopwords_filter": {
        "lang": "en",
    },
    "word_repetition_filter": {
        "lang": "en",
    },
    "llm_quality_score_filter": {
        "api_or_hf_model": "qwen2.5-72b-instruct",
        "min_score": 0.0,
        "enable_vllm": False,
    },
    "perplexity_filter": {
        "lang": "en",
    },
    "language_id_score_filter": {
        "lang": "en",
    },
    "llm_difficulty_score_filter": {
        "api_or_hf_model": "qwen2.5-72b-instruct",
        "min_score": 0.0,
        "enable_vllm": False,
    },
    # human annotators
    "human_preference_annotation_mapper": {
        "project_name_prefix": "Human_Preference_Annotation",
    },
}
