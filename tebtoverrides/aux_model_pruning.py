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
1. checken wie das obs["goal"] aussieht sollte jetzt geprintet werden
2. über prunner and die kummulierte token usage rankommen
3. und dann mal gucken was mit dem prompt noch so geht - aber eigentlich eh eine dumme idee

"""

from typing import Optional, List, Tuple, Dict, Any
import logging
import re
import os
from openai import OpenAI


import google.generativeai as genai


from dotenv import load_dotenv
load_dotenv()  # This loads .env file from current directory

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
        max_tokens: int = 4000,
        temperature: float = 0.0,
        enabled: bool = True,
        use_openrouter: bool = False,
        use_google_ai: bool = False
    ):
        """
        Initialize the auxiliary model pruner.
        
        Args:
            model_name: LLM model to use for pruning decisions
                       OpenAI: "gpt-4o-mini", "gpt-4o", "gpt-4-turbo"
                       OpenRouter: "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-exp", "deepseek/deepseek-chat-v3.1:free"
                       Google AI: "gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"
            max_tokens: Maximum tokens allowed in response
            temperature: Sampling temperature (0.0 for deterministic)
            enabled: Enable/disable pruning (for debugging)
            use_openrouter: If True, use OpenRouter API
            use_google_ai: If True, use Google AI Studio API (takes precedence over use_openrouter)
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.enabled = enabled
        self.use_openrouter = use_openrouter
        self.use_google_ai = use_google_ai
        
        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        
        # Initialize API client based on selection
        if use_google_ai:
            # Google AI Studio           
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)
            logger.info(f"✅ Google AI Studio client initialized: {model_name}")
            
        elif use_openrouter:
            # OpenRouter
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            logger.info(f"✅ OpenRouter client initialized: {model_name}")
        else:
            # OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.client = OpenAI(api_key=api_key)
            logger.info(f"✅ OpenAI client initialized: {model_name}")
    
    def call_llm(
        self,
        system_prompt: str,
        user_prompt: str
    ) -> Tuple[str, Dict[str, int]]:
        """
        Call the LLM API (OpenAI, OpenRouter, or Google AI) with system and user prompts.
        
        This method is optimized for repeated calls:
        - Reuses the same client instance (connection pooling)
        - Tracks token usage across all calls
        - Minimal overhead
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            
        Returns:
            Tuple of (llm_response_text, token_usage_dict)
            token_usage_dict contains: {
                "input_tokens": int,
                "output_tokens": int,
                "total_tokens": int
            }
            
        Raises:
            Exception: If API call fails
        """
        try:
            token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0
            }
            
            if self.use_google_ai:
                # Google AI Studio API call
                # Combine system and user prompts (Google doesn't have separate system prompt)
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                generation_config = genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                )
                
                response = self.client.generate_content(
                    full_prompt,
                    generation_config=generation_config
                )
                
                llm_response = response.text
                
                # Extract token usage from Google AI response
                if hasattr(response, 'usage_metadata'):
                    token_usage["input_tokens"] = response.usage_metadata.prompt_token_count
                    token_usage["output_tokens"] = response.usage_metadata.candidates_token_count
                    token_usage["total_tokens"] = response.usage_metadata.total_token_count
                
            else:
                # OpenAI or OpenRouter API call
                extra_params = {}
                if self.use_openrouter:
                    extra_params["extra_headers"] = {
                        "HTTP-Referer": "https://github.com/webmall-efficient",
                        "X-Title": "WebMall AXTree Pruning"
                    }
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **extra_params
                )
                
                llm_response = response.choices[0].message.content
                
                # Extract token usage from OpenAI/OpenRouter response
                if hasattr(response, 'usage') and response.usage:
                    token_usage["input_tokens"] = response.usage.prompt_tokens
                    token_usage["output_tokens"] = response.usage.completion_tokens
                    token_usage["total_tokens"] = response.usage.total_tokens
            
            # Update cumulative counters
            self.total_input_tokens += token_usage["input_tokens"]
            self.total_output_tokens += token_usage["output_tokens"]
            self.call_count += 1
            
            api_type = "Google AI" if self.use_google_ai else ("OpenRouter" if self.use_openrouter else "OpenAI")
            info = (
                f"📊 LLM Call #{self.call_count} ({api_type}) - Model: {self.model_name} | "
                f"Input: {token_usage['input_tokens']} | "
                f"Output: {token_usage['output_tokens']} | "
                f"Total: {token_usage['total_tokens']} tokens"
            )
            print(info)
            
            return llm_response, token_usage
            
        except Exception as e:
            logger.error(f"❌ Error calling LLM API: {e}")
            raise
    
    def get_token_stats(self) -> Dict[str, Any]:
        """
        Get cumulative token usage statistics.
        
        Returns:
            Dictionary with token usage stats:
            {
                "call_count": int,
                "total_input_tokens": int,
                "total_output_tokens": int,
                "total_tokens": int,
                "avg_input_tokens": float,
                "avg_output_tokens": float
            }
        """
        total = self.total_input_tokens + self.total_output_tokens
        avg_input = self.total_input_tokens / self.call_count if self.call_count > 0 else 0
        avg_output = self.total_output_tokens / self.call_count if self.call_count > 0 else 0
        
        return {
            "call_count": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": total,
            "avg_input_tokens": round(avg_input, 2),
            "avg_output_tokens": round(avg_output, 2)
        }
    
    def reset_token_stats(self):
        """Reset token usage counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        logger.info("🔄 Token statistics reset")

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
        logger.info("🔧 Created new global pruner instance")
    
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

# Goal:\n {goal}

# History : This is how the agent interacted with the task:\n{history}

# Observation:\n{axtree}

"""

def aux_model_pruning(
    axtree: str,
    goal: str,
    history_obs: Optional[List[dict]] = None,
    actions: Optional[List[str]] = None,
    memories: Optional[List[str]] = None,
    thoughts: Optional[List[str]] = None,
    system_prompt: str = default_system_prompt,
    user_prompt_template: str = default_prompt_template,
    model_name: str = "deepseek/deepseek-chat-v3.1:free",
    use_openrouter: bool = True,
    use_google_ai: bool = False,
    pruner: Optional[AuxModelPruner] = None  # ✅ NEU: Optional pruner parameter
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
        
    # Step 2: Format user prompt with template
    user_prompt = user_prompt_template.format(
        instruction=instruction,
        goal=goal,
        history=actions if actions else "No actions yet",
        axtree=numbered_axtree
    )
        
    # Step 3: Call LLM API
    llm_response, token_usage = pruner.call_llm(system_prompt, user_prompt)
        
    # Step 4: Extract pruned lines from LLM response
    pruned_lines = extract_pruned_lines(llm_response, original_lines)
    
    if pruned_lines is None:
        logger.warning("⚠️ Pruning failed, returning original AXTree")
        return axtree
        
    # Step 5: Return pruned AXTree
    pruned_axtree = "\n".join(pruned_lines)
    return pruned_axtree



