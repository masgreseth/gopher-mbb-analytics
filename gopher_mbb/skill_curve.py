import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# Set page config
st.set_page_config(
    page_title="Basketball Player Analytics",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# Get trend line coefficients for a player
def get_trend_line(df):
    """Returns slope and intercept of trend line, or None if not enough data"""
    if len(df) < 2:
        return None
    z = np.polyfit(df['%Ps'], df['ORtg'], 1)
    return z  # Returns [slope, intercept]

# Predict ORtg for a given %Ps
def predict_ortg(trend_coeffs, ps_value):
    """Predict ORtg given trend line coefficients and %Ps value"""
    if trend_coeffs is None:
        return None
    slope, intercept = trend_coeffs
    return slope * ps_value + intercept

# Calculate team ORtg
def calculate_team_ortg(player_trends, ps_allocations):
    """Calculate weighted team ORtg"""
    total = 0
    for player, ps in ps_allocations.items():
        if player in player_trends and player_trends[player] is not None:
            ortg = predict_ortg(player_trends[player], ps)
            if ortg is not None:
                total += (ps / 100) * ortg
    return total

# Load data function
@st.cache_data
def load_data():
    box_scores = []
    folder_path = 'gopher_mbb/box_scores'
    
    # Get all CSV files with their filenames
    csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # Extract opponent and date from filename
        # Format: opponent-MM-DD-YYYY.csv
        name_parts = filename.replace('.csv', '').split('-')
        if len(name_parts) >= 4:
            opponent = name_parts[0]
            date = f"{name_parts[1]}/{name_parts[2]}/{name_parts[3]}"
        else:
            opponent = filename.replace('.csv', '')
            date = "Unknown"
        
        df['opponent'] = opponent
        df['date'] = date
        df['game_label'] = f"{opponent.title()}\n{date}"
        
        box_scores.append(df)
    
    player_data = {}
    
    for df in box_scores:
        df = df[df['Name'] != 'TOTAL']
        
        for _, row in df.iterrows():
            player_name = row['Name']
            
            if player_name not in player_data:
                player_data[player_name] = []
            
            if pd.notna(row['ORtg']):
                player_data[player_name].append({
                    'ORtg': row['ORtg'],
                    '%Ps': row['%Ps'],
                    'opponent': row['opponent'],
                    'date': row['date'],
                    'game_label': row['game_label']
                })
    
    player_dfs = {}
    for player_name, data in player_data.items():
        player_dfs[player_name] = pd.DataFrame(data)
    
    return player_dfs

# Plot function
def create_player_plot(player_name, df):
    """Create enhanced plot for a player"""
    if df.empty:
        return None
    
    df = df.copy()
    df['Game'] = range(1, len(df) + 1)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Scatter plot without color gradient
    scatter = ax.scatter(df['%Ps'], df['ORtg'], s=150, alpha=0.6, 
                        color='steelblue', edgecolors='black', linewidth=1.5)
    
    # Add game labels (opponent and date)
    for idx, row in df.iterrows():
        ax.annotate(row['game_label'], (row['%Ps'], row['ORtg']),
                   textcoords="offset points", xytext=(0,10),
                   ha='center', fontsize=8, fontweight='normal',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3, edgecolor='none'))
    
    # Trend line
    if len(df) > 1:
        z = np.polyfit(df['%Ps'], df['ORtg'], 1)
        p = np.poly1d(z)
        ax.plot(df['%Ps'], p(df['%Ps']), "r--", alpha=0.8, linewidth=2,
               label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    # Labels and styling
    ax.set_xlabel('%Ps (Percentage of Possessions)', fontsize=13, fontweight='bold')
    ax.set_ylabel('ORtg (Offensive Rating)', fontsize=13, fontweight='bold')
    ax.set_title(f"{player_name}: Skill Curve Analysis", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    if len(df) > 1:
        ax.legend(fontsize=11)
    
    plt.tight_layout()
    return fig

# Player analysis page
def player_analysis_page(player_dfs):
    st.title("🏀 Player Skill Curve Analysis")

    if "show_skill_curve_help" not in st.session_state:
        st.session_state.show_skill_curve_help = False

    if st.button("📈 What is a Skill Curve?"):
        st.session_state.show_skill_curve_help = not st.session_state.show_skill_curve_help

    if st.session_state.show_skill_curve_help:
        st.info("""
        **What is a Skill Curve?**

        A skill curve models how a player's efficiency (ORtg) changes as their offensive role changes.

        Instead of assuming efficiency is constant, we estimate a curve using historical game data that captures three realities:

        **1. Diminishing Returns**  
        As a player takes on more possessions, defenses adjust. Shot quality declines and fatigue increases.

        **2. Role Optimization**  
        Every player has an optimal usage range where they are most efficient.  
        Below this range, they are underutilized. Above it, efficiency drops.

        **3. Lineup-Level Impact**  
        A team’s offensive rating is the weighted sum of individual efficiencies at their assigned roles.

        This model allows us to simulate coaching decisions:

        > “What happens if we give Player A 6% more possessions and Player B 6% fewer?”

        The optimizer then finds the possession distribution that maximizes total team ORtg.
        """)

    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Player Selection")
    st.sidebar.markdown("Select a player to view their skill curve")
    
    # Sort players by number of games (descending)
    player_games = {name: len(df) for name, df in player_dfs.items()}
    sorted_players = sorted(player_games.items(), key=lambda x: x[1], reverse=True)
    
    # Player selectbox with game count
    player_options = [f"{name} ({games} games)" for name, games in sorted_players]
    selected_option = st.sidebar.selectbox("Choose a player:", player_options)
    
    # Extract player name from selection
    selected_player = selected_option.split(" (")[0]
    
    # Display mode
    st.sidebar.markdown("---")
    show_stats = st.sidebar.checkbox("Show detailed statistics", value=True)
    show_data_table = st.sidebar.checkbox("Show game-by-game data", value=False)
    
    # Main content
    df = player_dfs[selected_player]
    
    # Player header
    st.header(f"📊 {selected_player}")
    
    if df.empty:
        st.warning("No data available for this player.")
        return
    
    # Key metrics in columns
    if show_stats:
        col1, col2, col3, col4 = st.columns(4)
        
        avg_ortg = df['ORtg'].mean()
        avg_ps = df['%Ps'].mean()
        games = len(df)
        correlation = df['ORtg'].corr(df['%Ps']) if len(df) > 1 else 0
        
        with col1:
            st.metric("Average ORtg", f"{avg_ortg:.1f}")
        with col2:
            st.metric("Average %Ps", f"{avg_ps:.1f}")
        with col3:
            st.metric("Games Played", games)
        with col4:
            st.metric("Correlation", f"{correlation:.3f}")
        
        st.markdown("---")
    
    # Plot
    fig = create_player_plot(selected_player, df)
    if fig:
        st.pyplot(fig)
        plt.close(fig)
    
    # Additional statistics
    if show_stats:
        st.markdown("### 📈 Performance Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**ORtg Statistics**")
            st.write(f"- Max: {df['ORtg'].max():.1f}")
            st.write(f"- Min: {df['ORtg'].min():.1f}")
            st.write(f"- Std Dev: {df['ORtg'].std():.1f}")
        
        with col2:
            st.markdown("**%Ps Statistics**")
            st.write(f"- Max: {df['%Ps'].max():.1f}")
            st.write(f"- Min: {df['%Ps'].min():.1f}")
            st.write(f"- Std Dev: {df['%Ps'].std():.1f}")
    
    # Game-by-game data table
    if show_data_table:
        st.markdown("---")
        st.markdown("### 📋 Game-by-Game Data")
        
        display_df = df.copy()
        display_df.insert(0, 'Game', range(1, len(display_df) + 1))
        # Reorder columns to show opponent and date first
        cols = ['Game', 'opponent', 'date', 'ORtg', '%Ps']
        display_df = display_df[cols]
        display_df = display_df.reset_index(drop=True)
        display_df.columns = ['Game', 'Opponent', 'Date', 'ORtg', '%Ps']
        
        st.dataframe(display_df, use_container_width=True)
    
    # Comparison mode
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Compare Players")
    if st.sidebar.checkbox("Enable comparison mode"):
        st.markdown("---")
        st.subheader("🔄 Player Comparison")
        
        other_players = st.multiselect(
            "Select players to compare:",
            [name for name in player_dfs.keys() if name != selected_player],
            default=[]
        )
        
        if other_players:
            fig_compare, ax = plt.subplots(figsize=(12, 7))
            
            # Plot selected player
            df_main = player_dfs[selected_player]
            ax.scatter(df_main['%Ps'], df_main['ORtg'], s=120, alpha=0.7,
                      label=selected_player, edgecolors='black', linewidth=1.5)
            
            # Plot comparison players
            for player in other_players:
                df_comp = player_dfs[player]
                ax.scatter(df_comp['%Ps'], df_comp['ORtg'], s=120, alpha=0.7,
                          label=player, edgecolors='black', linewidth=1.5)
            
            ax.set_xlabel('%Ps (Percentage of Possessions)', fontsize=13, fontweight='bold')
            ax.set_ylabel('ORtg (Offensive Rating)', fontsize=13, fontweight='bold')
            ax.set_title('Player Comparison: ORtg vs %Ps', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=11)
            
            plt.tight_layout()
            st.pyplot(fig_compare)
            plt.close(fig_compare)

# Get trend line coefficients for a player with penalty
def get_trend_line_with_penalty(df, penalty_threshold=None):
    """
    Returns a function that predicts ORtg with a penalty for extreme %Ps values.
    The penalty kicks in beyond the player's observed data range.
    """
    if len(df) < 2:
        return None
    
    # Get linear trend coefficients
    z = np.polyfit(df['%Ps'], df['ORtg'], 1)
    slope, intercept = z
    
    # Get the range of observed %Ps
    min_ps = df['%Ps'].min()
    max_ps = df['%Ps'].max()
    
    # If no penalty threshold provided, use max observed
    if penalty_threshold is None:
        penalty_threshold = max_ps
    
    # Calculate the ORtg at the penalty threshold (anchor point)
    anchor_ortg = slope * penalty_threshold + intercept
    
    def predict_with_penalty(ps_value):
        """
        Predict ORtg with exponential decay penalty beyond observed range.
        Always decreases after penalty threshold regardless of slope.
        """
        # If at or below threshold, return base linear prediction
        if ps_value <= penalty_threshold:
            return slope * ps_value + intercept
        
        # Beyond threshold, apply decay from anchor point
        excess = ps_value - penalty_threshold
        
        # Exponential decay - starts at anchor_ortg and decreases
        # The rate depends on how efficient the player is at threshold
        if slope > 0:
            # For positive slope (efficiency increases with usage)
            # Apply aggressive decay
            decay_rate = 0.12
            min_ortg = 50  # Floor value
        else:
            # For negative slope (efficiency already decreasing)
            # Apply moderate decay (compounding the existing decline)
            decay_rate = 0.08
            min_ortg = 40
        
        # Exponential decay from anchor point
        decayed_ortg = anchor_ortg * np.exp(-decay_rate * excess)
        
        # Add linear penalty component for steeper decline
        linear_penalty = excess * 1.5
        
        # Combine and apply floor
        final_ortg = decayed_ortg - linear_penalty
        
        return max(final_ortg, min_ortg)
    
    return {
        'slope': slope,
        'intercept': intercept,
        'min_ps': min_ps,
        'max_ps': max_ps,
        'penalty_threshold': penalty_threshold,
        'anchor_ortg': anchor_ortg,
        'predict': predict_with_penalty
    }

# Updated predict function
def predict_ortg(trend_data, ps_value):
    """Predict ORtg given trend data and %Ps value"""
    if trend_data is None:
        return None
    
    # If it's the new format with penalty
    if isinstance(trend_data, dict) and 'predict' in trend_data:
        return trend_data['predict'](ps_value)
    
    # Fallback to simple linear (for backwards compatibility)
    slope, intercept = trend_data
    return slope * ps_value + intercept

# Updated get_trend_line to use new model
def get_trend_line(df):
    """Returns trend line data with penalty model"""
    return get_trend_line_with_penalty(df)

# Team optimizer page with visualization of penalty zones
def team_optimizer_page(player_dfs):
    st.title("⚙️ Team Possession Optimizer")
    st.markdown("---")
    
    # Get players with enough data for trend lines
    player_trends = {}
    valid_players = []
    
    for player_name, df in player_dfs.items():
        trend = get_trend_line(df)
        if trend is not None:
            player_trends[player_name] = trend
            valid_players.append(player_name)
    
    if len(valid_players) == 0:
        st.error("No players have enough data for trend line analysis.")
        return
    
    # Sort players by average ORtg for better display
    player_avg_ortg = {}
    for player in valid_players:
        df = player_dfs[player]
        player_avg_ortg[player] = df['ORtg'].mean()
    
    valid_players = sorted(valid_players, key=lambda x: player_avg_ortg[x], reverse=True)
    
    # Sidebar - Player selection for optimizer
    st.sidebar.title("Optimizer Settings")
    st.sidebar.markdown("**Select Active Players:**")
    st.sidebar.caption("Deselect injured or unavailable players")
    
    # Create checkboxes for each player
    # if 'selected_players' not in st.session_state:
    #     st.session_state.selected_players = valid_players.copy()

    if "selected_players" not in st.session_state:
        injured_players = ["Robert Vaihola", "Chansey Willis"]
        st.session_state.selected_players = [
            p for p in valid_players if p not in injured_players
        ]

    
    # Select/Deselect all buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Select All"):
            st.session_state.selected_players = valid_players.copy()
            st.rerun()
    with col2:
        if st.button("❌ Clear All"):
            st.session_state.selected_players = []
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # Individual player checkboxes
    selected_players = []
    for player in valid_players:
        is_selected = st.sidebar.checkbox(
            player,
            value=player in st.session_state.selected_players,
            key=f"checkbox_{player}"
        )
        if is_selected and player not in ["Robert Vaihola", "Chansey Willis"]:
            selected_players.append(player)
    
    # Update session state
    st.session_state.selected_players = selected_players
    
    # Create prediction table
    st.header("📊 Predicted ORtg Table (All Players)")
    st.caption("⚠️ Predictions beyond a player's observed range include efficiency penalties")
    
    ps_increment = st.sidebar.slider("Table %Ps Increment:", 1, 5, 2)
    max_ps_table = st.sidebar.slider("Maximum %Ps for Table:", 20, 50, 45, 5)
    ps_values = list(range(0, max_ps_table + 1, ps_increment))
    
    # Build table for ALL valid players
    table_data = {}
    for player in valid_players:
        ortg_predictions = []
        for ps in ps_values:
            ortg = predict_ortg(player_trends[player], ps)
            ortg_predictions.append(round(ortg, 1) if ortg else None)
        table_data[player] = ortg_predictions
    
    pred_df = pd.DataFrame(table_data, index=ps_values)
    pred_df.index.name = '%Ps'
    
    # Style the dataframe
    st.dataframe(pred_df.style.background_gradient(cmap='RdYlGn', axis=None), 
                 use_container_width=True, height=400)
    
    # Show penalty zones for selected players
    if len(selected_players) > 0:
        with st.expander("📉 View Penalty Zones"):
            st.caption("Shows each player's observed range and where manual efficiency penalties begin (due to lack of data)")
            penalty_info = []
            for player in selected_players:
                trend = player_trends[player]
                penalty_info.append({
                    'Player': player,
                    'Min %Ps': f"{trend['min_ps']:.1f}",
                    'Max %Ps': f"{trend['max_ps']:.1f}",
                    'Penalty Starts': f"{trend['penalty_threshold']:.1f}",
                    'Slope': f"{trend['slope']:.2f}"
                })
            penalty_df = pd.DataFrame(penalty_info)
            st.dataframe(penalty_df, use_container_width=True, hide_index=True)
    
    # Add download button for the table
    csv = pred_df.to_csv()
    st.download_button(
        label="📥 Download Table as CSV",
        data=csv,
        file_name="player_ortg_predictions.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Only show optimizer if players are selected
    if len(selected_players) == 0:
        st.info("👆 Select players in the sidebar to use the lineup optimizer tool.")
        return
    
    # Show selected players count
    st.info(f"**Active Players ({len(selected_players)}):** {', '.join(selected_players)}")
    
    # Custom allocation tool
    st.header("🎯 Custom Possession Allocation")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Adjust Player Possessions")
        
        # Initialize session state for allocations
        if 'allocations' not in st.session_state:
            # Start with equal distribution
            equal_share = 100 / len(selected_players)
            st.session_state.allocations = {player: equal_share for player in selected_players}
        
        # Update allocations if player selection changed
        current_players = set(selected_players)
        stored_players = set(st.session_state.allocations.keys())
        
        if current_players != stored_players:
            # Add new players with equal share of remaining allocation
            existing_total = sum(st.session_state.allocations.get(p, 0) for p in current_players if p in stored_players)
            
            # Remove deselected players
            for player in stored_players - current_players:
                del st.session_state.allocations[player]
            
            # If there are new players, give them equal share
            new_players = current_players - stored_players
            if new_players:
                if existing_total < 100:
                    remaining = 100 - existing_total
                    share_per_new = remaining / len(new_players)
                    for player in new_players:
                        st.session_state.allocations[player] = share_per_new
                else:
                    # Need to normalize everything
                    equal_share = 100 / len(selected_players)
                    st.session_state.allocations = {player: equal_share for player in selected_players}
        
        # Create sliders for each player
        allocations = {}
        for i, player in enumerate(selected_players):
            # Show warning if beyond penalty threshold
            current_val = st.session_state.allocations.get(player, 0.0)
            penalty_thresh = player_trends[player]['penalty_threshold']
            
            label = f"{player}"
            if current_val > penalty_thresh:
                label += f" ⚠️ (Beyond {penalty_thresh:.0f}% - efficiency penalty applied)"
            
            allocations[player] = st.slider(
                label,
                min_value=0.0,
                max_value=100.0,
                value=float(current_val),  # Ensure it's a float
                step=0.5,
                key=f"slider_{player}_{current_val}"  # Add current_val to key to force update
            )
        
        # Update session state only if values changed
        st.session_state.allocations = allocations
        
        # Show total
        total_ps = sum(allocations.values())
        if abs(total_ps - 100) < 0.1:
            st.success(f"✅ Total: {total_ps:.1f}%")
        else:
            st.error(f"⚠️ Total: {total_ps:.1f}% (Must equal 100%)")
        
        # Action buttons
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            if st.button("⚖️ Normalize to 100%"):
                if total_ps > 0:
                    factor = 100 / total_ps
                    for player in selected_players:
                        st.session_state.allocations[player] = allocations[player] * factor
                    st.rerun()
        with button_col2:
            if st.button("🔄 Equal Distribution"):
                equal_share = 100 / len(selected_players)
                for player in selected_players:
                    st.session_state.allocations[player] = equal_share
                st.rerun()
    
    with col2:
        st.subheader("Results")
        
        # Calculate current team ORtg
        if abs(total_ps - 100) < 0.1:
            team_ortg = calculate_team_ortg(player_trends, allocations)
            st.metric("Team ORtg", f"{team_ortg:.2f}")
            
            st.markdown("---")
            st.markdown("**Individual Contributions:**")
            
            contributions = []
            for player in selected_players:
                ps = allocations[player]
                ortg = predict_ortg(player_trends[player], ps)
                contribution = (ps / 100) * ortg
                
                # Add penalty indicator
                penalty_thresh = player_trends[player]['penalty_threshold']
                player_display = player if ps <= penalty_thresh else f"{player} ⚠️"
                
                contributions.append({
                    'Player': player_display,
                    '%Ps': f"{ps:.1f}%",
                    'ORtg': f"{ortg:.1f}",
                    'Contribution': f"{contribution:.2f}"
                })
            
            contrib_df = pd.DataFrame(contributions)
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Adjust allocations to equal 100%")
    
    st.markdown("---")
    
    # Optimization section
    st.header("🚀 Find Optimal Allocation")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        min_ps = st.number_input("Minimum %Ps per player:", 0.0, 50.0, 0.0, 1.0)
    with col2:
        max_ps = st.number_input("Maximum %Ps per player:", 0.0, 100.0, 35.0, 1.0)
    with col3:
        st.write("")  # Spacing
        optimize_button = st.button("🎯 Optimize Lineup", type="primary")
    
    if optimize_button:
        # Optimization function
        def objective(x):
            """Negative team ORtg (we minimize, so negate to maximize)"""
            alloc = {player: x[i] for i, player in enumerate(selected_players)}
            return -calculate_team_ortg(player_trends, alloc)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 100}  # Sum to 100
        ]
        
        # Bounds
        bounds = [(min_ps, max_ps) for _ in selected_players]
        
        # Initial guess
        x0 = np.array([100 / len(selected_players)] * len(selected_players))
        
        # Optimize
        with st.spinner("Finding optimal allocation..."):
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_alloc = {player: result.x[i] for i, player in enumerate(selected_players)}
            optimal_ortg = calculate_team_ortg(player_trends, optimal_alloc)
            
            st.success(f"✅ Optimal Team ORtg: {optimal_ortg:.2f}")
            
            # Display optimal allocation
            optimal_data = []
            for player in selected_players:
                ps = optimal_alloc[player]
                ortg = predict_ortg(player_trends[player], ps)
                contribution = (ps / 100) * ortg
                
                # Add penalty indicator
                penalty_thresh = player_trends[player]['penalty_threshold']
                player_display = player if ps <= penalty_thresh else f"{player} ⚠️"
                
                optimal_data.append({
                    'Player': player_display,
                    '%Ps': f"{ps:.1f}%",
                    'Predicted ORtg': f"{ortg:.1f}",
                    'Contribution': f"{contribution:.2f}"
                })
            
            optimal_df = pd.DataFrame(optimal_data)
            st.dataframe(optimal_df, use_container_width=True, hide_index=True)
            
            # Button to apply optimal allocation
            # if st.button("📋 Apply Optimal Allocation"):
            #     st.session_state.allocations = optimal_alloc
            #     st.session_state.allocation_version += 1  # INCREMENT VERSION
            #     st.rerun()
        else:
            st.error("Optimization failed. Try adjusting the min/max constraints.")

# Main app
def main():
    # Load data
    with st.spinner('Loading player data...'):
        player_dfs = load_data()
    
    if not player_dfs:
        st.error("No data found. Please make sure the 'box_scores' folder contains CSV files.")
        return
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Player Analysis", "⚙️ Team Optimizer"])
    
    with tab1:
        player_analysis_page(player_dfs)
    
    with tab2:
        team_optimizer_page(player_dfs)

if __name__ == "__main__":

    main()

