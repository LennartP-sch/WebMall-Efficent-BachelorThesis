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
        data: List of tuples (model_name, avg_cost, completion_rate, color, marker, edgecolor)
              Example: [("AX-Tree (GPT)", 0.30, 56.0, "red", "o", "black"), ...]
              edgecolor: "white" for normal, "green" for pruned
        title: Plot title
        output_file: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Collect legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Plot each model
    for item in data:
        if len(item) == 6:
            model_name, avg_cost, completion_rate, color, marker, edgecolor = item
        else:
            # Fallback für alte Daten ohne edgecolor
            model_name, avg_cost, completion_rate, color, marker = item
            edgecolor = 'white'
        
        size = 100 if "Memory" in model_name or "Vision" in model_name else 150
        scatter = ax.scatter(avg_cost, completion_rate, s=size, c=color, marker=marker, 
                   alpha=0.7, edgecolors=edgecolor, linewidth=2.5)
        
        # Add to legend
        legend_handles.append(scatter)
        legend_labels.append(model_name)
    
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
    ax.set_ylim(0, 100)
    
    # Add grid
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend with all models (Symbol + Name)
    ax.legend(legend_handles, legend_labels, 
              loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, 
              frameon=True, fancybox=True, shadow=True,
              title='Models (Green edge = Pruned)', title_fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")
    
    # Show plot
    plt.show()


def create_f1_token_plot(data, title, output_file):
    """
    Creates a scatter plot showing F1-Score vs Average Tokens Used.
    
    Args:
        data: List of tuples (model_name, avg_tokens, f1_score, color, marker, edgecolor)
              Example: [("AX-Tree (GPT)", 150000, 0.56, "red", "o", "white"), ...]
              edgecolor: "white" for normal, "green" for pruned, "yellow" for pruned2
        title: Plot title
        output_file: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Collect legend handles and labels
    legend_handles = []
    legend_labels = []
    
    # Plot each model
    for item in data:
        if len(item) == 6:
            model_name, avg_tokens, f1_score, color, marker, edgecolor = item
        else:
            # Fallback für alte Daten ohne edgecolor
            model_name, avg_tokens, f1_score, color, marker = item
            edgecolor = 'white'
        
        size = 100 if "Memory" in model_name or "Vision" in model_name else 150
        scatter = ax.scatter(avg_tokens, f1_score, s=size, c=color, marker=marker, 
                   alpha=0.7, edgecolors=edgecolor, linewidth=2.5)
        
        # Add to legend
        legend_handles.append(scatter)
        legend_labels.append(model_name)
    
    # Set axis properties
    ax.set_xlabel('Average Tokens Used per Task', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set x-axis range
    ax.set_xlim(0, 600000)
    
    # Format x-axis ticks with thousands separator
    from matplotlib.ticker import FuncFormatter
    def token_formatter(x, pos):
        if x >= 1000:
            return f'{x/1000:.0f}K'
        else:
            return f'{x:.0f}'
    
    ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
    
    # Set y-axis range (F1-Score is 0-1)
    ax.set_ylim(0, 1.0)
    
    # Add grid
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add legend with all models (Symbol + Name)
    ax.legend(legend_handles, legend_labels, 
              loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9, 
              frameon=True, fancybox=True, shadow=True,
              title='Models (Yellow = Pruned2)', title_fontsize=10)
    
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
    Add your models here with format: (name, cost, completion_rate, color, marker, edgecolor)
    edgecolor: 'white' for standard, 'green' for pruned
    """
    
    # Example data structure - Replace with your actual data
    basic_tasks_data = [
        ("AX-Tree (GPT 4.1)", 0.28, 56.25, "#E06666", "o", "white"),
        ("AX-Tree (Claude Sonnet 4)", 0.67, 66.67, "#E06666", "s", "white"),
        ("AX-Tree + Memory (GPT 4.1)", 0.29, 75.0, "#3D85C6", "o", "white"),
        ("AX-Tree + Memory (Claude Sonnet 4)", 0.94, 70.83, "#3D85C6", "s", "white"),
        ("AX-Tree + Vision (GPT 4.1)", 0.29, 56.25, "#6AA84F", "o", "white"),
        ("AX-Tree + Vision (Claude Sonnet 4)", 0.82, 72.92, "#6AA84F", "s", "white"),
        ("Vision (GPT 4.1)", 0.23, 41.67, "#F8B166", "o", "white"),
        ("Vision (Claude Sonnet 4)", 1.3, 10.42, "#F6B26B", "s", "white"),
    ]
    
    # ✅ ADD YOUR MODELS HERE
    #Avg steps: 24.625
    basic_tasks_data.append(("AX-Tree + Memory (GROK 4 FAST)", 0.016293, 79.16, "#3D85C6", "D", "white"))
    #Avg steps: 18.9583
    basic_tasks_data.append(("AX-Tree + Memory (Gemini 2.5 flash)", 0.0387, 6.25, "#3D85C6", "^", "white"))
    #Avg steps: 22.625
    basic_tasks_data.append(("PRUNED1 AX-Tree + Memory (GROK 4 FAST)", 0.013764, 83.33, "#3D85C6", "D", "green"))
    #Avg steps: 22.7916  
    basic_tasks_data.append(("PRUNED2 AX-Tree + Memory (Gemini 2.5 flash)", 0.03250, 16.666, "#3D85C6", "^", "yellow"))

    #Avg steps: 22.4166
    basic_tasks_data.append(("HTML (Gemini 2.5 flash)", 0.075421, 47.91, "#DD56FF", "^", "white"))
    #Avg steps: 24.77083
    basic_tasks_data.append(("PRUNED2 HTML (Gemini 2.5 flash)", 0.1440, 35.41, "#DD56FF", "^", "yellow")) 

    if not basic_tasks_data:
        print("⚠️  No data added for Basic Tasks plot!")
        print("Add your models in the basic_tasks_data list.")
        return
    
    output_file = Path(__file__).parent / "cost_vs_completion_basic_tasks.png"
    create_cost_completion_plot(
        data=basic_tasks_data,
        title="Cost vs Completion Rate: Basic Tasks",
        output_file=output_file
    )


def create_advanced_tasks_plot():
    """
    Create plot for Advanced Tasks.
    Add your models here with format: (name, cost, completion_rate, color, marker, edgecolor)
    edgecolor: 'white' for standard, 'green' for pruned
    """
    
    # Example data structure - Replace with your actual data
    advanced_tasks_data = [
        ("AX-Tree (GPT 4.1)", 0.35, 32.56, "#E06666", "o", "white"),
        ("AX-Tree (Claude Sonnet 4)", 1.02, 53.49, "#E06666", "s", "white"),
        ("AX-Tree + Memory (GPT 4.1)", 0.40, 34.88, "#3D85C6", "o", "white"),
        ("AX-Tree + Memory (Claude Sonnet 4)", 1.37, 48.84, "#3D85C6", "s", "white"),
        ("AX-Tree + Vision (GPT 4.1)", 0.36, 39.53, "#6AA84F", "o", "white"),
        ("AX-Tree + Vision (Claude Sonnet 4)", 1.63, 37.21, "#6AA84F", "s", "white"),
        ("Vision (GPT 4.1)", 0.29, 13.95, "#F8B166", "o", "white"),
        ("Vision (Claude Sonnet 4)", 1.53, 4.65, "#F6B26B", "s", "white"),
    ]
    
    # ✅ ADD YOUR MODELS HERE
    #Avg steps: 28.8
    advanced_tasks_data.append(("AX-Tree + Memory (GROK 4 FAST)", 0.103591, 45.71, "#3D85C6", "D", "white"))
    #Avg steps: 19.942
    advanced_tasks_data.append(("AX-Tree + Memory (Gemini 2.5 flash)", 0.0515, 2.85, "#3D85C6", "^", "white")) 
    #Avg steps: 28.8
    advanced_tasks_data.append(("PRUNED1 AX-Tree + Memory (GROK 4 FAST)", 0.082449, 51.428, "#3D85C6", "D", "green"))
    #Avg steps: 30.74285  
    advanced_tasks_data.append(("PRUNED2 AX-Tree + Memory (GROK 4 FAST)", 0.084958, 57.142, "#3D85C6", "D", "yellow"))
    #Avg steps: 17.8857  
    advanced_tasks_data.append(("PRUNED2 AX-Tree + Memory (Gemini 2.5 flash)", 0.0361, 5.714, "#3D85C6", "^", "yellow"))

    #Avg steps: 27.2857
    advanced_tasks_data.append(("HTML (Gemini 2.5 flash)", 0.12441, 14.28, "#DD56FF", "^", "white"))
    #Avg steps: 31.9714
    advanced_tasks_data.append(("HTML PRUNED1 (Gemini 2.5 flash)", 0.11858, 8.57, "#DD56FF", "^", "green"))
    #Avg steps: 28.057142 
    advanced_tasks_data.append(("HTML PRUNED2 (Gemini 2.5 flash)", 0.1883, 14.28, "#DD56FF", "^", "yellow"))  


    if not advanced_tasks_data:
        print("⚠️  No data added for Advanced Tasks plot!")
        print("Add your models in the advanced_tasks_data list.")
        return
    
    output_file = Path(__file__).parent / "cost_vs_completion_advanced_tasks.png"
    create_cost_completion_plot(
        data=advanced_tasks_data,
        title="Cost vs Completion Rate: Advanced Tasks",
        output_file=output_file
    )


def create_basic_tasks_f1_plot():
    """
    Create F1-Score vs Tokens plot for Basic Tasks.
    Add your models here with format: (name, avg_tokens, f1_score, color, marker, edgecolor)
    edgecolor: 'white' for standard, 'green' for pruned1, 'yellow' for pruned2
    """
    
    # ✅ ADD YOUR DATA HERE
    # Format: (model_name, avg_tokens, f1_score, color, marker, edgecolor)
    basic_tasks_f1_data = [
        # Example - replace with your actual data:       
        ("AX-Tree + Memory (GROK 4 FAST)", 278700.14 + 21438.25, 0.83778712, "#3D85C6", "D", "white"),
        ("AX-Tree + Memory (GPT 4.1)", 130270.875 +  3511.187, 0.87607, "#3D85C6", "o", "white"),
        ("AX-Tree + Memory (Claude Sonnet 4)",236631.25 + 15106.875 , 0.780648 , "#3D85C6", "s", "white")
        
    ]
    
    if not basic_tasks_f1_data:
        print("⚠️  No data added for Basic Tasks F1-Score plot!")
        print("Add your models in the basic_tasks_f1_data list.")
        print("Format: (model_name, avg_tokens, f1_score, color, marker, edgecolor)")
        return
    
    output_file = Path(__file__).parent / "f1_vs_tokens_basic_tasks.png"
    create_f1_token_plot(
        data=basic_tasks_f1_data,
        title="F1-Score vs Average Tokens Used: Basic Tasks",
        output_file=output_file
    )


def create_advanced_tasks_f1_plot():
    """
    Create F1-Score vs Tokens plot for Advanced Tasks.
    Add your models here with format: (name, avg_input_tokens + avg_output_tokens, f1_score, color, marker, edgecolor)
    edgecolor: 'white' for standard, 'green' for pruned1, 'yellow' for pruned2
    """
    
    # ✅ ADD YOUR DATA HERE
    # Format: (model_name, avg_tokens, f1_score, color, marker, edgecolor)
    advanced_tasks_f1_data = [
        ("AX-Tree + Memory (GROK 4 FAST)", 436720.9 + 32494.5, 0.6012, "#3D85C6", "D", "white"),
        ("PRUNED2 AX-Tree + Memory (GROK 4 FAST)", 343229.8 + 32625.54, 0.65414, "#3D85C6", "D", "yellow"),
        ("AX-Tree + Memory (GPT 4.1)", 178949.116 + 4658.13, 0.49006, "#3D85C6", "o", "white"),
        ("AX-Tree + Memory (Claude Sonnet 4)",364858.8372 + 18149.279 ,0.59914 , "#3D85C6", "s", "white")
    ]
    
    if not advanced_tasks_f1_data:
        print("⚠️  No data added for Advanced Tasks F1-Score plot!")
        print("Add your models in the advanced_tasks_f1_data list.")
        print("Format: (model_name, avg_tokens, f1_score, color, marker, edgecolor)")
        return
    
    output_file = Path(__file__).parent / "f1_vs_tokens_advanced_tasks.png"
    create_f1_token_plot(
        data=advanced_tasks_f1_data,
        title="F1-Score vs Average Tokens Used: Advanced Tasks",
        output_file=output_file
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Creating Cost vs Completion Rate Plots & F1-Score vs Tokens Plots")
    print("=" * 80)
    
    # Create Basic Tasks plot
    print("\n📊 Creating Basic Tasks Cost vs Completion plot...")
    create_basic_tasks_plot()
    
    # Create Advanced Tasks plot
    print("\n📊 Creating Advanced Tasks Cost vs Completion plot...")
    create_advanced_tasks_plot()
    
    # Create Basic Tasks F1-Score plot
    print("\n📊 Creating Basic Tasks F1-Score vs Tokens plot...")
    create_basic_tasks_f1_plot()
    
    # Create Advanced Tasks F1-Score plot
    print("\n📊 Creating Advanced Tasks F1-Score vs Tokens plot...")
    create_advanced_tasks_f1_plot()
    
    print("\n" + "=" * 80)
    print("✅ Done! Add your model data in the functions above.")
    print("=" * 80)