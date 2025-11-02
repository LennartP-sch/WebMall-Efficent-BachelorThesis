"""
Auxiliary Model Pruning for AXTree Optimization

This module provides context-aware pruning of accessibility trees using a secondary language model.
The pruning considers the agent's goal, previous observations, actions, memories, and thoughts
to intelligently remove irrelevant elements and reduce token count.

Performance optimizations:
- Singleton pattern for API clients (reuse connections)
- Connection pooling via OpenAI client
- Minimal overhead for repeated calls

Supported APIs:
- OpenAI (GPT-4, GPT-4o-mini, etc.)
- OpenRouter (Claude, Gemini, Llama, DeepSeek, etc.)
- Google AI Studio (Gemini 2.5 Flash, Gemini 1.5 Pro, etc.)


SOTA -2k Tokens ist halt mega wenig
TODO
1. test machen

"""

import json
import os
import logging
import re
from typing import Optional, Dict, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)


class AuxModelPruner:
    """
    Context-aware AXTree pruner using an auxiliary language model.
    
    Performance optimized with:
    - Reusable API client (singleton pattern)
    - Connection pooling
    - Token usage tracking
    
    Supports three API types:
    - OpenAI: use_openrouter=False, use_google_ai=False
    - OpenRouter: use_openrouter=True, use_google_ai=False  
    - Google AI Studio: use_google_ai=True
    
    Attributes:
        model_name: Name of the LLM to use for pruning decisions
        max_tokens: Maximum tokens for the pruned output
        temperature: Temperature for model sampling
        enabled: Whether pruning is enabled
        use_openrouter: Whether to use OpenRouter API
        use_google_ai: Whether to use Google AI Studio API
        client: API client instance (reused across calls)
        total_input_tokens: Cumulative input tokens used
        total_output_tokens: Cumulative output tokens used
        call_count: Number of API calls made
    """
    
    def __init__(
        self,
        model_name: str = "deepseek/deepseek-chat-v3.1:free",
        use_openrouter: bool = True,
        use_google_ai: bool = False,
        system_prompt: str = """\
Your are part of a web agent who's job is to solve a task. Your are
currently at a step of the whole episode, and your job is to extract the
relevant information for solving the task. An agent will execute the task
after you on the subset that you extracted. Make sure to extract sufficient
information to be able to solve the task, but also remove information
that is irrelevant to reduce the size of the observation and all the distractions.
""",
        user_prompt_template: str = """{instruction}

# Goal:\n {goal}

# Observation:\n{axtree}

"""
    ):
        """
        Initialize the auxiliary model pruner.
        
        Args:
            model_name: LLM model to use for pruning decisions
                       OpenAI: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"
                       OpenRouter: "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp", "deepseek/deepseek-chat-v3.1:free"
                       Google AI: "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"
            use_openrouter: If True, use OpenRouter API
            use_google_ai: If True, use Google AI Studio API (takes precedence over use_openrouter)
            system_prompt: System prompt for the LLM
            user_prompt_template: User prompt template
        """
        self.model_name = model_name
        self.use_openrouter = use_openrouter
        self.use_google_ai = use_google_ai
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

        # Initialize counters
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # ✅ NEU: Stats-Datei für Ray Multi-Processing
        self.stats_file = Path(os.getenv('AGENTLAB_EXP_ROOT', '.')) / 'aux_pruning_stats_temp.json'

        # Initialize API clients
        if self.use_google_ai:
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_2nd_KEY")
                if not api_key:
                    raise ValueError("GEMINI_2nd_KEY environment variable not set")
                genai.configure(api_key=api_key)
                self.genai_client = genai.GenerativeModel(model_name)
                logger.info(f"[OK] Google AI Studio client initialized: {model_name}")  # ✅ Kein Emoji
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize Google AI client: {e}")
                raise
        elif self.use_openrouter:
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError("OPENROUTER_API_KEY environment variable not set")
                self.openai_client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                logger.info(f"[OK] OpenRouter client initialized: {model_name}")  # ✅ Kein Emoji
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize OpenRouter client: {e}")
                raise
        else:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI()
                logger.info(f"[OK] OpenAI client initialized: {model_name}")  # ✅ Kein Emoji
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize OpenAI client: {e}")
                raise

    def call_llm(self, system_prompt: str, user_prompt: str) -> Tuple[str, Dict[str, int]]:
        """
        Call the configured LLM API and return the response with token usage.
        """
        try:
            if self.use_google_ai:
                # Google AI API call
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.genai_client.generate_content(full_prompt)
                llm_response = response.text

                # Extract token usage from Google AI
                token_usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }

            else:
                # OpenAI/OpenRouter API call
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                )

                llm_response = response.choices[0].message.content

                # Extract token usage
                token_usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            # Update cumulative counters
            self.total_input_tokens += token_usage["input_tokens"]
            self.total_output_tokens += token_usage["output_tokens"]
            self.call_count += 1

            api_type = "Google AI" if self.use_google_ai else ("OpenRouter" if self.use_openrouter else "OpenAI")
            
            # ✅ GEÄNDERT: Keine Emojis mehr!
            logger.info(
                f"[LLM] Call #{self.call_count} ({api_type}) - Model: {self.model_name} | "
                f"Input: {token_usage['input_tokens']} | "
                f"Output: {token_usage['output_tokens']} | "
                f"Total: {token_usage['total_tokens']} tokens"
            )

            # ✅ NEU: Schreibe Stats nach jedem Call
            self._save_stats_to_file()

            return llm_response, token_usage

        except Exception as e:
            logger.error(f"[ERROR] Error calling LLM API: {e}")  # ✅ Kein Emoji
            raise

    def _save_stats_to_file(self):
        """Save current stats to JSON file (Windows-compatible, no fcntl)."""
        stats = self.get_token_stats()
        stats['pid'] = os.getpid()  # Process ID für Ray Workers

        try:
            # ✅ Windows-kompatibel: Verwende einfaches File I/O
            # Ray isoliert die Worker, also ist Race Condition unwahrscheinlich
            
            # Lese existierende Stats
            all_stats = {}
            if self.stats_file.exists():
                try:
                    with open(self.stats_file, 'r') as f:
                        all_stats = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass

            # Update Stats für diesen Worker
            worker_key = f"worker_{os.getpid()}"
            all_stats[worker_key] = stats

            # Schreibe zurück (atomar mit temp file)
            temp_file = self.stats_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(all_stats, f, indent=2)
            
            # Atomic rename (Windows-kompatibel)
            temp_file.replace(self.stats_file)

        except Exception as e:
            # ✅ Kein Emoji
            logger.warning(f"[WARNING] Could not save pruning stats to file: {e}")

    def get_token_stats(self) -> Dict[str, float]:
        """Return cumulative token usage statistics."""
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "avg_input_tokens": (
                self.total_input_tokens / self.call_count if self.call_count > 0 else 0
            ),
            "avg_output_tokens": (
                self.total_output_tokens / self.call_count if self.call_count > 0 else 0
            ),
        }

def number_axtree_lines(axtree: str) -> Tuple[str, List[str]]:
    """
    Add line numbers to AXTree and return both versions.
    
    Args:
        axtree: Original AXTree string
        
    Returns:
        Tuple of (numbered_axtree_string, original_lines_list)
    """
    original_lines = axtree.splitlines()
    numbered_lines = []
    
    for idx, line in enumerate(original_lines, start=1):
        numbered_lines.append(f"{idx} {line}")

    return "\n".join(numbered_lines), original_lines


def extract_pruned_lines(response: str, original_lines: List[str]) -> List[str]:
    """
    Extract specific lines from the original AXTree based on LLM response.
    
    The LLM response should contain line numbers/ranges within <answer> tags.
    Supports ONLY formats:
    - Single line in tuple: (5)
    - Range in tuple: (10, 12) means lines 10-12 inclusive
    - List of tuples: [(1), (3, 5), (8)]
    
    Args:
        response: LLM response containing <answer>...</answer> tags
        original_lines: List of original AXTree lines (0-indexed)
        
    Returns:
        List of selected lines from original AXTree
        
    Example:
        >>> response = "<answer>\\n[(10, 12), (5), (123, 124)]\\n</answer>"
        >>> original = ["line1", "line2", ..., "line124"]
        >>> result = extract_pruned_lines(response, original)
        >>> # Returns line 5, lines 10-12, and lines 123-124
    """
    # Extract content between <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    
    if not answer_match:
        print("Auxiliary model pruning failed: No <answer> tags found.")
        return original_lines
    
    answer_content = answer_match.group(1).strip()
    
    # Parse the line numbers/ranges
    line_indices = set()
    
    try:
        # Find all tuples: either (x, y) for ranges or (x) for single lines
        # Pattern matches: (number) or (number, number)
        tuple_pattern = r'\((\d+)(?:\s*,\s*(\d+))?\)'
        tuple_matches = re.findall(tuple_pattern, answer_content)
        
        for match in tuple_matches:
            start_str, end_str = match
            start = int(start_str)
            
            if end_str:  # Range format: (start, end)
                end = int(end_str)
                # Convert 1-indexed to 0-indexed and add all lines in range
                for i in range(start - 1, end):  # end is inclusive
                    if 0 <= i < len(original_lines):
                        line_indices.add(i)
            else:  # Single line format: (number)
                # Convert 1-indexed to 0-indexed
                idx = start - 1
                if 0 <= idx < len(original_lines):
                    line_indices.add(idx)
        
        if not line_indices:
            return None
        
        # Sort indices and extract corresponding lines
        sorted_indices = sorted(line_indices)
        result_lines = [original_lines[i] for i in sorted_indices]
        
        original_count = len(original_lines)
        pruned_count = len(result_lines)
        reduction = ((original_count - pruned_count) / original_count * 100) if original_count > 0 else 0

        print(f"Pruned AXTree with LLM: {reduction:.1f}% reduction")
        
        return result_lines
        
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}. Returning None.")
        return None

# Global pruner instance (singleton pattern for performance)
_global_pruner: Optional[AuxModelPruner] = None


def get_pruner(
    model_name: str = "deepseek/deepseek-chat-v3.1:free",
    use_openrouter: bool = False,
    use_google_ai: bool = False,
    **kwargs 
) -> AuxModelPruner:
    """
    Get or create the global pruner instance.
    
    This implements the singleton pattern to reuse the API client connection
    across multiple calls, which is crucial for performance when aux_model_pruning
    is called repeatedly.
    
    Args:
        model_name: LLM model name
        use_openrouter: Whether to use OpenRouter API
        use_google_ai: Whether to use Google AI Studio API (takes precedence)
        **kwargs: Additional arguments for AuxModelPruner
        
    Returns:
        Global AuxModelPruner instance (reused across calls)
    """
    global _global_pruner
    
    if _global_pruner is None:
        _global_pruner = AuxModelPruner(
            model_name=model_name,
            use_openrouter=use_openrouter,
            use_google_ai=use_google_ai,
            **kwargs
        )
    
    return _global_pruner

default_system_prompt = """\
Your are part of a web agent who's job is to solve a task. Your are
currently at a step of the whole episode, and your job is to extract the
relevant information for solving the task. An agent will execute the task
after you on the subset that you extracted. Make sure to extract sufficient
information to be able to solve the task, but also remove information
that is irrelevant to reduce the size of the observation and all the distractions.
"""

abstract_ex = """
# Abstract example
Here is an abstract example of how your answer should be formatted:
<think>
Reason about which lines of the AxTree should be kept to achieve the goal specified in # Goal.
</think>
<answer>
A list of line numbers ranges that are relevant to achieve the goal. For example: [(10,12), (123, 456)]
</answer>
"""

concrete_ex = """
# Concrete example
Here is an example of what your answer should look like:
<think>
The lines that are relevant to achieve the goal are:
- Line 10 to 12: This line contains the information about the product price.
- Line 123 : This line contains the add to cart button.
</think>
<answer>
[(10,12), (123)]
</answer>
"""
instruction = """\
# Instructions
Extract the lines that can be relevant for the task at this step of completion.
A final AXTree will be built from these lines. It should contain enough information to understand the state of the shopping page,
the current step and to perform the right next action, including buttons, links and any element to interact with.
Returning less information then needed leads to task failure. Make sure to return enough information.
Golden Rules:
- Be extensive and not restrictive. It is always better to return more lines rather than less.
- If unsure whether a line is relevant, keep it.
Expected answer format:
<think>
Reason about which lines of the AxTree should be kept to achieve the goal specified in # Goal.
</think>
<answer>
A list of line numbers ranges that are relevant to achieve the goal. For example: [(10,12), (123)]
</answer>
"""

default_prompt_template = """{instruction}

{aexample}

{cexample}

# Goal:\n {goal}

# Agent's last reasoning \n{message}

# Observation \n{axtree}
"""
## History : This is how the agent interacted with the task:\n{history}
#Laut FocusAgent Paper reduced das Information
# The url of this page : \n{url}


def aux_model_pruning(
    axtree: str,
    goal: str,
    url: str,
    chatmessages,  # ← Chat Historie (nicht nützlich)
    actions=None,  # ✅ NEU: Bisherige Actions
    thoughts=None,  # ✅ NEU: Bisherige Thoughts/Reasonings
    model_name: str = "deepseek/deepseek-chat-v3.1:free",
    use_openrouter: bool = True,
    use_google_ai: bool = False,
    pruner: Optional[AuxModelPruner] = None,
    full_obs: dict = None
) -> str:
    """
    Main function to prune AXTree using auxiliary LLM via REST API.
    
    This function is called multiple times in quick succession from dynamic_prompting.py,
    so it's optimized for performance:
    - Uses singleton pattern for API client (reuses connections)
    - Connection pooling via OpenAI client
    - Minimal overhead between calls
    
    Supports three API types:
    - OpenAI: use_openrouter=False, use_google_ai=False
    - OpenRouter: use_openrouter=True, use_google_ai=False (default)
    - Google AI Studio: use_google_ai=True
    
    Args:
        axtree: Full accessibility tree string
        goal: Agent's current goal/task
        history_obs: List of previous observations
        actions: List of previous actions taken
        memories: List of previous memories
        thoughts: List of previous thoughts/reasoning
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template
        model_name: LLM model to use
        use_openrouter: If True, use OpenRouter
        use_google_ai: If True, use Google AI Studio (takes precedence)
        
    Returns:
        Pruned AXTree string
    """
    # ✅ Nutze übergebenen Pruner oder hole globalen
    if pruner is None:
        pruner = get_pruner(
            model_name=model_name,
            use_openrouter=use_openrouter,
            use_google_ai=use_google_ai
        )
        
    # Step 1: Number the AXTree lines
    numbered_axtree, original_lines = number_axtree_lines(axtree)

    task_match = re.search(r'<task>(.*?)</task>', goal, re.DOTALL | re.IGNORECASE)    
    goal = task_match.group(1).strip() if task_match else goal

    # ✅ DEBUG: Print Actions und Thoughts
    #print("\n" + "=" * 100)
    #print("🔍 ACTIONS & THOUGHTS DEBUG")
    #print("=" * 100)
    
    #print(f"\n📊 Actions type: {type(actions)}")
    #print(f"📊 Actions length: {len(actions) if actions else 0}")
    
    #if actions:
    #    print("\n📋 ACTIONS:")
    #    for idx, action in enumerate(actions[-5:]):  # Letzte 5 Actions
    #        print(f"  [{idx}] {action}")
    
    #print(f"\n📊 Thoughts type: {type(thoughts)}")
    #print(f"📊 Thoughts length: {len(thoughts) if thoughts else 0}")
    
    #if thoughts:
    #    print("\n💭 THOUGHTS (last 3):")
    #    for idx, thought in enumerate(thoughts[-3:]):  # Letzte 3 Thoughts
    #        print(f"\n  Thought #{idx + 1}:")
    #       print(f"  {thought[:500]}...")  # Erste 500 chars
    
    #print("=" * 100 + "\n")

    # ✅ Verwende letztes Thought oder letzte Action
    #last_context = ""
    
    if thoughts and len(thoughts) > 0:
        # Verwende letztes Reasoning
        last_context = thoughts[-1]
    else:
        last_context = "No previous reasoning"

    if actions and len(actions) > 0:
        last_action = actions[-1]
    else:
        last_action = "No actions taken"    

    # Step 2: Format user prompt
    user_prompt = default_prompt_template.format(
        instruction=instruction,
        aexample=abstract_ex,
        cexample=concrete_ex,
        goal=goal,
        #url=url,
        message=last_context,  # ✅ Verwende Thought oder Action
        axtree=numbered_axtree
    )
        
    # Step 3: Call LLM API
    llm_response, token_usage = pruner.call_llm(default_system_prompt, user_prompt)
        
    # Step 4: Extract pruned lines
    pruned_lines = extract_pruned_lines(llm_response, original_lines)
    
    if pruned_lines is None:
        print("LLM pruning failed. Returned AXTree ohne Änderungen")
        return axtree
        
    # Step 5: Return pruned AXTree
    pruned_axtree = "\n".join(pruned_lines)
    return pruned_axtree



