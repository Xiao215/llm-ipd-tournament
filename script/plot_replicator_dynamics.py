import numpy as np
import matplotlib.pyplot as plt
from src.evolution.replicator_dynamics import Population_Payoffs, Discrete_Replicator_Dynamics


def plot_probability_evolution(trajectory, colors=None, labels=None, figsize=(10, 6)):
    """
    Plot the evolution of a probability distribution over time as a stacked area chart.
    
    Parameters:
    -----------
    trajectory : list of numpy arrays
        Each array is a probability distribution at a given time step.
        Each array should be shape (n,) and sum to 1.
    colors : list, optional
        Colors for each item. If None, uses default matplotlib colormap
    labels : list, optional  
        Labels for each item. If None, uses "Item 0", "Item 1", etc.
    figsize : tuple
        Figure size (width, height)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """
    
    # Convert trajectory to 2D array: (time_steps, n_items)
    prob_matrix = np.array(trajectory)
    n_time_steps, n_items = prob_matrix.shape
    
    # Validate that probabilities sum to 1 (approximately)
    sums = np.sum(prob_matrix, axis=1)
    if not np.allclose(sums, 1.0, rtol=1e-3):
        print(f"Warning: Some probability distributions don't sum to 1. Sums range from {sums.min():.4f} to {sums.max():.4f}")
    
    # Compute cumulative sums for each time step
    cumsum_matrix = np.cumsum(prob_matrix, axis=1)
    
    # Add a column of zeros at the beginning for the base
    cumsum_with_base = np.column_stack([np.zeros(n_time_steps), cumsum_matrix])
    
    # Create time axis
    time_axis = np.arange(n_time_steps)
    
    # Set up colors
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, n_items))
    
    # Set up labels
    if labels is None:
        labels = [f'Item {i}' for i in range(n_items)]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked areas using fill_between
    for i in range(n_items):
        ax.fill_between(time_axis, 
                       cumsum_with_base[:, i], 
                       cumsum_with_base[:, i+1], 
                       color=colors[i], 
                       label=labels[i],
                       alpha=0.8,
                       edgecolor='white',
                       linewidth=0.5)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Evolution of Probability Distribution Over Time')
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax

def plot_share_progression(pop_payoff, dynamics_results):
    """
    Plot the evolution of a population distribution over time as a stacked area chart
    with overlaid payoff trajectory.
    
    Parameters:
    -----------
    pop_payoff : an instance of Population_Payoffs
    dynamics_results: contains the following:
        trajectory : list of numpy arrays
            Each array is a population distribution at a given time step.
        pay_traj : list of float
            Payoff values at each time step.
        status : status of the dynamics
            
    Returns:
    --------
    fig, (ax1, ax2) : matplotlib figure and axes objects
        ax1: primary axis for population shares
        ax2: secondary axis for payoff trajectory
    """
    pop_traj, pay_traj, status = dynamics_results


    
    # Convert trajectory to 2D array and transpose for stackplot
    prob_matrix = np.array(pop_traj).T  # Shape: (n_types, n_time_steps)
    
    # Get indices that would sort the final population in descending order
    sorted_indices = np.argsort( -prob_matrix[:, -1] )  # Negative to sort in descending order
    
    # Reorder prob_matrix rows based on final population values
    prob_matrix = prob_matrix[sorted_indices]
    
    # Also reorder the labels accordingly
    labels = [pop_payoff.agent_types[i] for i in sorted_indices]

    n_types, n_time_steps = prob_matrix.shape
    
    # Create time axis
    time_axis = np.arange(n_time_steps)
    
    # Set up colors
    colors = plt.cm.Set3(np.linspace(0, 1, n_types))
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot stacked areas for population shares
    ax1.stackplot(time_axis, *prob_matrix, colors=colors, labels=labels, alpha=0.7)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Share of Population', color='black')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, n_time_steps - 1)
    
    # Add a thick horizontal line at y=1
    ax1.axhline(y=1, color='black', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for payoff trajectory
    ax2 = ax1.twinx()
    payoff_line = ax2.plot(time_axis, pay_traj, color='red', linewidth=3, 
                          label='Average Payoff', alpha=0.9)
    ax2.set_ylabel('Average Population Payoff', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    # Set y-axis limits for the payoff plot based on min/max values in the payoff tensor
    ax2.set_ylim(pop_payoff.payoff_tensor.min(), pop_payoff.payoff_tensor.max())
    # Set title
    ax1.set_title('Evolution of Population Shares and Average Population Payoff Over Time')

    # Add invisible line with status as label (using circle marker)
    status_line = ax1.plot([], [], 'o', markersize=5, markerfacecolor='white', 
                          markeredgecolor='black', label=f'Status: {status}')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              bbox_to_anchor=(1.15, 1), loc='upper left')
    
    plt.tight_layout()
    return fig, (ax1, ax2)


# Example usage and demonstration
if False:
    # Generate example trajectory simulating algorithm performance over time
    np.random.seed(42)
    n_time_steps = 50
    n_items = 4
    
    # Create a trajectory where probabilities shift over time
    # This could represent different algorithms' relative performance
    trajectory = []
    for t in range(n_time_steps):
        # Start with base probabilities
        base_probs = np.array([0.4, 0.3, 0.2, 0.1])
        
        # Add time-dependent variation (simulate changing conditions)
        variation = 0.15 * np.array([
            np.sin(2 * np.pi * t / 25),           # Algorithm A: periodic
            -np.sin(2 * np.pi * t / 25) * 0.7,    # Algorithm B: inverse periodic
            0.1 * np.cos(2 * np.pi * t / 15),     # Algorithm C: different frequency
            np.exp(-t/30) * 0.2                   # Algorithm D: decay over time
        ])
        
        probs = base_probs + variation
        
        # Ensure probabilities are non-negative and sum to 1
        probs = np.maximum(probs, 0.01)  # Minimum probability
        probs = probs / np.sum(probs)    # Normalize
        
        trajectory.append(probs)
    
    # Define colors and labels for the example
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    labels = ['Deep Q-Network', 'Policy Gradient', 'Actor-Critic', 'Monte Carlo Tree Search']
    
    # Create both versions of the plot
    print("Creating stacked area chart using fill_between method...")
    fig1, ax1 = plot_probability_evolution(trajectory, colors=colors, labels=labels)
    plt.show()
    
    print("\nCreating stacked area chart using stackplot method...")
    fig2, ax2 = plot_probability_evolution_stackplot(trajectory, colors=colors, labels=labels)
    plt.show()
    
    # Example with your own trajectory data:
    # Uncomment and modify the following lines to use your own data
    # 
    # your_trajectory = [prob_dist_1, prob_dist_2, ...]  # List of numpy arrays
    # your_colors = ['red', 'blue', 'green', 'yellow']   # Optional
    # your_labels = ['Method A', 'Method B', ...]        # Optional
    # 
    # fig, ax = plot_probability_evolution(your_trajectory, 
    #                                    colors=your_colors, 
    #                                    labels=your_labels)
    # plt.show()

def plot_3simplex_trajectories(trajectories):
    """Plot trajectories on the 3-strategy simplex"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw simplex triangle  
    triangle = Polygon([(0, 0), (1, 0), (0.5, np.sqrt(3)/2)], 
                        fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # Convert simplex coordinates to triangle coordinates
    def simplex_to_triangle(p0, p1, p2):
        x = p1 + p2/2
        y = p2 * np.sqrt(3)/2
        return x, y
    
    # Plot trajectories
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, trajectory in enumerate(trajectories):
        x_coords, y_coords = [], []
        for state in trajectory:
            x, y = simplex_to_triangle(state[0], state[1], state[2])
            x_coords.append(x)
            y_coords.append(y)
        
        ax.plot(x_coords, y_coords, color=colors[i % len(colors)], 
                alpha=0.7, linewidth=2, label=f'Trajectory {i+1}')
        ax.plot(x_coords[0], y_coords[0], 'o', color=colors[i % len(colors)], 
            markersize=8)
        ax.plot(x_coords[-1], y_coords[-1], 's', color=colors[i % len(colors)], 
            markersize=8)
        
    ax.plot([], [], 'o', color='black', markersize=8, label='Start')
    ax.plot([], [], 's', color='black', markersize=8, label='End')  

    # Labels for corners
    ax.text(-0.05, -0.05, 'Type 0', fontsize=12, ha='center')
    ax.text(1.05, -0.05, 'Type 1', fontsize=12, ha='center')  
    ax.text(0.5, np.sqrt(3)/2 + 0.05, 'Type 2', fontsize=12, ha='center')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Evolutionary Game Dynamics', 
                fontsize=14, fontweight='bold')

   
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig, ax

# Example usage and experiments
def run_experiments(experiment):

    """
    Create a 3-player, 3-action game: Cooperate (C), Defect (D), Punish (P)
    
    Game logic:
    - Cooperate: Benefits when others cooperate, exploited by defectors
    - Defect: Exploits cooperators, suffers against punishers
    - Punish: Specifically targets defectors, costly but effective
    """
    
    # Payoff tensor: payoff_tensor[i,j,k] = payoff to player 1 when players play (i,j,k)
    # Actions: 0=Cooperate, 1=Defect, 2=Punish
    payoff_tensor = np.array([
        # Player 1 plays Cooperate (0)
        [
            [3.0, 2.5, 2.0],  # Others: (C,C), (C,D), (C,P)
            [1.0, 0.5, 1.5],  # Others: (D,C), (D,D), (D,P)  
            [2.5, 1.5, 2.0]   # Others: (P,C), (P,D), (P,P)
        ],
        # Player 1 plays Defect (1)  
        [
            [4.0, 3.5, 3.0],  # Others: (C,C), (C,D), (C,P)
            [2.0, 1.5, 1.0],  # Others: (D,C), (D,D), (D,P)
            [0.5, 0.0, 0.5]   # Others: (P,C), (P,D), (P,P) - punished!
        ],
        # Player 1 plays Punish (2)
        [
            [1.8, 2.8, 2.3],  # Others: (C,C), (C,D), (C,P) 
            [3.2, 2.2, 2.7],  # Others: (D,C), (D,D), (D,P) - punishing defectors
            [1.5, 2.0, 1.8]   # Others: (P,C), (P,D), (P,P)
        ]
    ])
    
    action_names = ['Cooperate', 'Defect', 'Punish']

    pop_payoffs = Population_Payoffs(agent_types=action_names, payoff_tensor=payoff_tensor)
    RD = Discrete_Replicator_Dynamics(pop_payoffs,  payoffs_updating=False)

    if experiment == "simplex_plot":
        trajectories = []
        for i in range(5):  # Run 5 different trajectories
            pops, pays, status  = RD.run_mw_dynamics(initial_population="random", steps=10000)
            print(f"Trajectory {i+1} status: {status}")
            trajectories.append(pops)  # Store only the population history

        # Create visualization
        fig, ax = plot_3simplex_trajectories(trajectories)
        plt.show()
    elif experiment == "population_progression":
        # Run the dynamics
        dynamics_results = RD.run_mw_dynamics(initial_population="uniform", steps=500)
        print(f"Experiment status: {dynamics_results[-1]}")

        # Plot the population progression
        fig, ax = plot_share_progression(pop_payoffs, dynamics_results)
        plt.show()
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

if True:
    run_experiments("population_progression")