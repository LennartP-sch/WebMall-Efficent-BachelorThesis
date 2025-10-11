"""
Script to create Cost vs Completion Rate plots for WebMall study results.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_cost_completion_plot(data, title, output_file):
    """
    Creates a scatter plot showing Cost vs Completion Rate.
    
    Args:
        data: List of tuples (model_name, avg_cost, completion_rate, color, marker)
              Example: [("AX-Tree (GPT)", 0.30, 56.0, "red", "o"), ...]
        title: Plot title
        output_file: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each model
    for model_name, avg_cost, completion_rate, color, marker in data:
        size = 100 if "Memory" in model_name or "Vision" in model_name else 150
        ax.scatter(avg_cost, completion_rate, s=size, c=color, marker=marker, 
                   alpha=0.7, edgecolors='white', linewidth=1.5, label=model_name)
    
    # Set axis properties
    ax.set_xlabel('Average Cost per Task ($) - Log Scale', fontsize=12, fontweight='bold')
    ax.set_ylabel('Task Completion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis to log scale with dollar formatting
    ax.set_xscale('log')
    ax.set_xlim(0.01, 2.0)  # From $0.01 to $2.00
    
    # Format x-axis ticks as dollar values
    from matplotlib.ticker import FuncFormatter
    def dollar_formatter(x, pos):
        if x >= 1:
            return f'${x:.2f}'
        elif x >= 0.1:
            return f'${x:.2f}'
        else:
            return f'${x:.3f}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(dollar_formatter))
    
    # Set specific x-axis tick locations for better readability
    ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
    
    # Set y-axis range
    ax.set_ylim(0, 80)
    
    # Add grid
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, 
              frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    
    # Show plot
    plt.show()


def create_basic_tasks_plot():
    """
    Create plot for Basic Tasks.
    Add your models here with format: (name, cost, completion_rate, color, marker)
    """
    
    # Example data structure - Replace with your actual data
    basic_tasks_data = [
        ("AX-Tree (GPT 4.1)", 0.28, 56.25, "#E06666", "o"),
        ("AX-Tree (Claude Sonnet 4)", 0.67, 66.67, "#E06666", "s"),
        ("AX-Tree + Memory (GPT 4.1)", 0.29, 75.0, "#3D85C6", "o"),
        ("AX-Tree + Memory (Claude Sonnet 4)", 0.94, 70.83, "#3D85C6", "s"),
        ("AX-Tree + Vision (GPT 4.1)", 0.29, 56.25, "#6AA84F", "o"),
        ("AX-Tree + Vision (Claude Sonnet 4)", 0.82, 72.92, "#6AA84F", "s"),
        ("Vision (GPT 4.1)", 0.23, 41.67, "#F8B166", "o"),
        ("Vision (Claude Sonnet 4)", 1.3, 10.42, "#F6B26B", "s"),
    ]
    
    # ✅ ADD YOUR MODELS HERE
    # Example:
    basic_tasks_data.append(("AX-Tree + Memory (GROK 4 FAST)", 0.016293, 79.16, "#3D85C6", "D"))
    basic_tasks_data.append(("AX-Tree + Memory (Gemini 2.5 flash)", 0.0387, 6.25, "#3D85C6", "^"))
    basic_tasks_data.append(("HTML (Gemini 2.5 flash)", 0.075421, 47.91, "#DD56FF", "^"))
    basic_tasks_data.append(("PRUNED HTML (Gemini 2.5 flash)", 0.064678, 45.83, "#DD56FF", "^")) 
    #15% cost reduction with pruning, 5% drop in performance
    
    if not basic_tasks_data:
        print("⚠️  No data added for Basic Tasks plot!")
        print("Add your models in the basic_tasks_data list.")
        return
    
    output_file = Path("cost_vs_completion_basic_tasks.png")
    create_cost_completion_plot(
        data=basic_tasks_data,
        title="Cost vs Completion Rate: Basic Tasks",
        output_file=output_file
    )


def create_advanced_tasks_plot():
    """
    Create plot for Advanced Tasks.
    Add your models here with format: (name, cost, completion_rate, color, marker)
    """
    
    # Example data structure - Replace with your actual data
    advanced_tasks_data = [
        ("AX-Tree (GPT 4.1)", 0.35, 32.56, "#E06666", "o"),
        ("AX-Tree (Claude Sonnet 4)", 1.02, 53.49, "#E06666", "s"),
        ("AX-Tree + Memory (GPT 4.1)", 0.40, 34.88, "#3D85C6", "o"),
        ("AX-Tree + Memory (Claude Sonnet 4)", 1.37, 48.84, "#3D85C6", "s"),
        ("AX-Tree + Vision (GPT 4.1)", 0.36, 39.53, "#6AA84F", "o"),
        ("AX-Tree + Vision (Claude Sonnet 4)", 1.63, 37.21, "#6AA84F", "s"),
        ("Vision (GPT 4.1)", 0.29, 13.95, "#F8B166", "o"),
        ("Vision (Claude Sonnet 4)", 1.53, 4.65, "#F6B26B", "s"),
    ]
    
    # ✅ ADD YOUR MODELS HERE
    # Example:
    advanced_tasks_data.append(("AX-Tree + Memory (GROK 4 FAST)", 0.0249, 45.71, "#3D85C6", "D"))
    advanced_tasks_data.append(("AX-Tree + Memory (Gemini 2.5 flash)", 0.0515, 3.0, "#3D85C6", "^"))
    advanced_tasks_data.append(("HTML (Gemini 2.5 flash)", 0.12441, 14.28, "#DD56FF", "^"))
    advanced_tasks_data.append(("HTML PRUNED (Gemini 2.5 flash)", 0.11858, 8.57, "#DD56FF", "^"))
    
    if not advanced_tasks_data:
        print("⚠️  No data added for Advanced Tasks plot!")
        print("Add your models in the advanced_tasks_data list.")
        return
    
    output_file = Path("cost_vs_completion_advanced_tasks.png")
    create_cost_completion_plot(
        data=advanced_tasks_data,
        title="Cost vs Completion Rate: Advanced Tasks",
        output_file=output_file
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Creating Cost vs Completion Rate Plots")
    print("=" * 80)
    
    # Create Basic Tasks plot
    print("\n📊 Creating Basic Tasks plot...")
    create_basic_tasks_plot()
    
    # Create Advanced Tasks plot
    print("\n📊 Creating Advanced Tasks plot...")
    create_advanced_tasks_plot()
    
    print("\n" + "=" * 80)
    print("✅ Done! Add your model data in the functions above.")
    print("=" * 80)