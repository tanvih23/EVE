#for seeing what walle is seeing, has 2 modes- ascii map and matplotlib plots
#also prints the final mission report to the terminal and saves it to a json file

import numpy as np


# ascii map

def ascii_map(env, agent_pos, particle_filter, step_num,
              battery, strategy='', show_belief=True):
    #draw the current state as an ascii map
    density = particle_filter.density_map(env.height, env.width)
    estimate = particle_filter.estimate()

    lines = []

    # header
    lines.append(f"═══ Step {step_num:3d} │ Battery: {battery:3d} │ "
                 f"Strategy: {strategy:8s} │ "
                 f"Confidence: {particle_filter.confidence():.1%} ═══")
    lines.append(f"    Estimate: ({estimate[0]:.1f}, {estimate[1]:.1f}) │ "
                 f"Entropy: {particle_filter.grid_entropy():.2f} bits │ "
                 f"N_eff: {particle_filter.effective_sample_size():.0f}")
    lines.append("")

    # column numbers
    col_header = "    "
    for c in range(env.width):
        col_header += f"{c % 10}"
    lines.append(col_header)
    lines.append("    " + "─" * env.width)

    # grid
    belief_chars = " ░▒▓█"  # density levels

    for r in range(env.height):
        row_str = f"{r:2d} │ "
        for c in range(env.width):
            # priority: agent > eve > clue > obstacle > recharge > belief
            if (r, c) == tuple(agent_pos):
                row_str += "W"
            elif (r, c) == env.eve_pos and (r, c) == tuple(agent_pos):
                row_str += "★"
            elif env.grid[r, c] == 1:
                row_str += "█"
            elif env.grid[r, c] == 2:
                row_str += "⚡"
            elif (r, c) in env.clues:
                if env.clues[(r, c)]['found']:
                    row_str += "!"
                else:
                    row_str += "·"  # agent cant see clues it hasnt found
            elif show_belief and density[r, c] > 0.001:
                # map density to a shade char
                level = min(4, int(density[r, c] * 100))
                row_str += belief_chars[level]
            else:
                row_str += "·"
        lines.append(row_str)

    lines.append("")
    return "\n".join(lines)


def print_mission_report(report):
    #pretty-print the mission report to the terminal
    lines = []
    summary = report['mission_summary']
    eve = report['eve_location']

    lines.append("")
    lines.append("╔══════════════════════════════════════════════╗")
    lines.append("║         WALL-E MISSION REPORT                ║")
    lines.append("╚══════════════════════════════════════════════╝")
    lines.append("")

    # outcome
    outcome = summary['outcome']
    if outcome == 'EVE_FOUND':
        lines.append("  ★ MISSION SUCCESS: EVE LOCATED! ★")
    elif outcome == 'MAX_STEPS':
        lines.append("  ⊘ Mission ended: maximum steps reached")
    else:
        lines.append("  ⊘ Mission ended: battery depleted")

    lines.append("")
    lines.append(f"  Steps taken:      {summary['steps_taken']}/{summary['max_steps']}")
    lines.append(f"  Battery remaining: {summary['battery_remaining']}/{summary['battery_max']}")
    lines.append(f"  Cells visited:    {summary['cells_visited']}")
    lines.append(f"  Time elapsed:     {summary['elapsed_seconds']:.3f}s")

    lines.append("")
    lines.append("  ── EVE Location Estimate ──")
    est = eve['particle_filter_estimate']
    lines.append(f"  Particle filter:  ({est[0]:.2f}, {est[1]:.2f})")
    lines.append(f"  Confidence:       {eve['confidence']:.2%}")
    lines.append(f"  Entropy:          {eve['entropy_bits']:.2f} bits")

    if eve['triangulation_estimate']:
        tri = eve['triangulation_estimate']
        lines.append(f"  Triangulation:    ({tri[0]:.2f}, {tri[1]:.2f})")

    lines.append("")
    lines.append("  ── Top Candidate Cells ──")
    for i, cell_info in enumerate(eve['top_candidate_cells']):
        cell = cell_info['cell']
        prob = cell_info['probability']
        bar = "█" * int(prob * 50)
        lines.append(f"  {i+1}. ({cell[0]:2d}, {cell[1]:2d})  {prob:.1%}  {bar}")

    lines.append("")
    lines.append("  ── Evidence ──")
    ev = report['evidence']
    lines.append(f"  Clues found:      {ev['clues_found']}")
    lines.append(f"  Signal readings:  {ev['signal_readings']}")

    lines.append("")
    lines.append("  ── Strategy Breakdown ──")
    strat = report['strategy_analysis']
    for name, count in strat['strategy_counts'].items():
        pct = count / max(strat['total_decisions'], 1) * 100
        lines.append(f"  {name:10s}: {count:3d} steps ({pct:.0f}%)")

    lines.append("")
    lines.append(f"  ── Recommended Next Step ──")
    lines.append(f"  {report['recommended_next_step']}")
    lines.append("")

    return "\n".join(lines)


# matplotlib charts

def plot_mission_analysis(trace, report, env, particle_filter,
                          output_path='output/analysis.png'):
    #the 2x2 mission plot: belief heatmap+path, confidence over time, entropy
    #over time, and the strategy timeline. saved to output_path.
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        print("Install with: pip install matplotlib")
        return None

    # pull the step-level data out of the trace
    steps = []
    confidences = []
    entropies = []
    strategies = []
    positions_r = []
    positions_c = []

    for entry in trace:
        if 'step' in entry and 'confidence' in entry:
            steps.append(entry['step'])
            confidences.append(entry['confidence'])
            entropies.append(entry['entropy'])
            strategies.append(entry.get('strategy', 'unknown'))
            positions_r.append(entry['position'][0])
            positions_c.append(entry['position'][1])

    if not steps:
        print("No step data to plot.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WALL-E Mission Analysis", fontsize=16, fontweight='bold')

    # 1) belief heatmap
    ax1 = axes[0, 0]
    density = particle_filter.density_map(env.height, env.width)

    # dark -> bright green colormap (wall-e theme)
    colors = ['#0a0a0a', '#0d3d0d', '#1a7a1a', '#33cc33', '#66ff66']
    cmap = LinearSegmentedColormap.from_list('walle', colors)

    im = ax1.imshow(density, cmap=cmap, interpolation='nearest',
                    aspect='auto')
    fig.colorbar(im, ax=ax1, label='Belief density')

    # obstacles + chargers
    for r in range(env.height):
        for c in range(env.width):
            if env.grid[r, c] == 1:
                ax1.plot(c, r, 's', color='#444', markersize=4)
            elif env.grid[r, c] == 2:
                ax1.plot(c, r, 's', color='#2a6', markersize=5, alpha=0.7)

    # decoys = orange triangles
    for decoy in getattr(env, 'decoys', []):
        dr, dc = decoy['position']
        ax1.plot(dc, dr, '^', color='#ff9a3c', markersize=9, mew=1,
                 markeredgecolor='black', label='_decoy')
    # clues = diamonds, filled if found, hollow if still hidden
    for (cr, cc), info in getattr(env, 'clues', {}).items():
        if info.get('found'):
            ax1.plot(cc, cr, 'D', color='#ffd24a', markersize=5, alpha=0.9,
                     label='_cluef')
        else:
            ax1.plot(cc, cr, 'D', markersize=5, markerfacecolor='none',
                     markeredgecolor='#ffd24a', alpha=0.6, label='_clueh')

    # agent path
    ax1.plot(positions_c, positions_r, '-', color='white',
             linewidth=0.8, alpha=0.5)
    ax1.plot(positions_c[0], positions_r[0], 's',
             color='#9ad', markersize=7, label='Start')
    ax1.plot(positions_c[-1], positions_r[-1], 'o',
             color='cyan', markersize=8, label='WALL-E')
    ax1.plot(env.eve_pos[1], env.eve_pos[0], '*',
             color='red', markersize=14, label='EVE (true)')

    estimate = particle_filter.estimate()
    ax1.plot(estimate[1], estimate[0], 'x',
             color='yellow', markersize=10, mew=2, label='Estimate')

    # one proxy handle each for decoys/clues so the legend stays tidy
    from matplotlib.lines import Line2D
    extra = []
    if getattr(env, 'decoys', []):
        extra.append(Line2D([0], [0], marker='^', color='none',
                     markerfacecolor='#ff9a3c', markeredgecolor='black',
                     markersize=9, label='Decoy (spoof)'))
    if getattr(env, 'clues', {}):
        extra.append(Line2D([0], [0], marker='D', color='none',
                     markerfacecolor='#ffd24a', markersize=6, label='Clue'))
    ax1._extra_handles = extra

    ax1.set_title('Belief, Path, Clues & Decoys')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    handles, labels = ax1.get_legend_handles_labels()
    handles += getattr(ax1, '_extra_handles', [])
    ax1.legend(handles=handles, loc='upper right', fontsize=7)

    # 2) confidence over time
    ax2 = axes[0, 1]
    ax2.plot(steps, confidences, color='#33cc33', linewidth=1.5)
    ax2.axhline(y=0.70, color='orange', linestyle='--',
                alpha=0.7, label='Exploit threshold')
    ax2.fill_between(steps, confidences, alpha=0.15, color='#33cc33')
    ax2.set_title('Confidence Over Time')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Confidence')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3) entropy over time
    ax3 = axes[1, 0]
    ax3.plot(steps, entropies, color='#ff6633', linewidth=1.5)
    ax3.fill_between(steps, entropies, alpha=0.15, color='#ff6633')
    ax3.set_title('Entropy Over Time (lower = more certain)')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Entropy (bits)')
    ax3.grid(True, alpha=0.3)

    # 4) strategy timeline
    ax4 = axes[1, 1]
    strategy_colors = {
        'explore': '#3399ff',
        'exploit': '#33cc33',
        'retreat': '#ff3333',
    }
    for i, (step, strat) in enumerate(zip(steps, strategies)):
        color = strategy_colors.get(strat, '#999999')
        ax4.barh(0, 1, left=step, color=color, height=0.8)

    # dashed line and label wherever the strategy flips so we exploit
    prev = None
    for step, strat in zip(steps, strategies):
        if strat != prev:
            ax4.axvline(step, color='white', linestyle=':', alpha=0.5, lw=0.8)
            ax4.text(step, 0.55, strat, rotation=90, fontsize=6,
                     color='white', va='bottom', ha='center', alpha=0.8)
            prev = strat

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3399ff', label='Explore (info gain)'),
        Patch(facecolor='#33cc33', label='Exploit (move to target)'),
        Patch(facecolor='#ff3333', label='Retreat (recharge)'),
    ]
    ax4.legend(handles=legend_elements, loc='lower right', fontsize=8)
    ax4.set_ylim(-0.5, 1.2)
    ax4.set_title('Strategy Timeline (dashed = strategy change)')
    ax4.set_xlabel('Step')
    ax4.set_yticks([])
    ax4.set_xlim(0, max(steps) + 1)

    plt.tight_layout()

    # save
    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()

    return output_path


# benchmark charts

def plot_benchmark(results, output_path='output/benchmark/benchmark.png'):
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("WALL-E Benchmark — Generalisation, Robustness, Coordination",
                 fontsize=15, fontweight='bold')

    pcolors = {
        'epistemic': '#33cc33', 'belief_greedy': '#3399ff',
        'greedy': '#ffb13c', 'lawnmower': '#aa66dd', 'random': '#ff5566',
    }

    # 1) success rate by planner & difficulty
    ax = axes[0, 0]
    comp = results['planner_comparison']
    diffs = list(comp.keys())
    planners = list(next(iter(comp.values())).keys())
    x = np.arange(len(diffs))
    width = 0.8 / max(len(planners), 1)
    for i, name in enumerate(planners):
        vals = [comp[d][name]['success_rate'] for d in diffs]
        ax.bar(x + i * width, vals, width, label=name,
               color=pcolors.get(name, '#999'))
    ax.set_xticks(x + width * (len(planners) - 1) / 2)
    ax.set_xticklabels(diffs)
    ax.set_ylabel('Success rate')
    ax.set_ylim(0, 1)
    ax.set_title('Success Rate by Planner & Difficulty')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis='y', alpha=0.3)

    # 2) median steps to find
    ax = axes[0, 1]
    ref_diff = 'easy' if 'easy' in comp else diffs[0]
    names, steps_vals = [], []
    for name in planners:
        s = comp[ref_diff][name]['median_steps_to_find']
        names.append(name)
        steps_vals.append(s if s is not None else 0)
    bars = ax.bar(names, steps_vals,
                  color=[pcolors.get(n, '#999') for n in names])
    for b, v in zip(bars, steps_vals):
        if v == 0:
            ax.text(b.get_x() + b.get_width() / 2, 1, 'n/a',
                    ha='center', fontsize=7, color='#ccc')
    ax.set_ylabel('Median steps to find EVE')
    ax.set_title(f'Efficiency on "{ref_diff}" missions (lower = better)')
    ax.tick_params(axis='x', rotation=30, labelsize=7)
    ax.grid(True, axis='y', alpha=0.3)

    # 3) robustness curves
    ax = axes[1, 0]
    rob = results['robustness']
    ns = rob['noise_sweep']
    dc = rob['decoy_sweep']
    ax.plot([p['noise'] for p in ns], [p['success_rate'] for p in ns],
            '-o', color='#33cc33', label='vs sensor noise (0 decoys)')
    ax.plot([p['decoys'] for p in dc], [p['success_rate'] for p in dc],
            '-s', color='#ff9a3c', label='vs decoy count (noise 0.8)')
    ax.set_ylabel('Success rate')
    ax.set_xlabel('Sensor-noise σ  /  decoy count')
    ax.set_ylim(0, 1)
    ax.set_title('Robustness: Degradation under Noise & Decoys')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4) multi-agent scaling
    ax = axes[1, 1]
    ma = results['multiagent']
    sizes = sorted(int(k) for k in ma.keys())
    srs = [ma[str(k) if str(k) in ma else k]['success_rate'] for k in sizes]
    ticks = [ma[str(k) if str(k) in ma else k]['median_ticks_to_find'] or 0
             for k in sizes]
    ax.bar([s - 0.15 for s in sizes], srs, 0.3, color='#33cc33',
           label='success rate')
    ax.set_ylabel('Success rate', color='#33cc33')
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    ax2.plot(sizes, ticks, '-o', color='#3399ff', label='ticks to find')
    ax2.set_ylabel('Median ticks to find', color='#3399ff')
    ax.set_xlabel('Number of coordinated units')
    ax.set_xticks(sizes)
    ax.set_title('Multi-Agent Scaling (shared belief)')

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    return output_path


def plot_multiagent(ma_report, env, output_path='output/multiagent.png'):
    #plot each unit's path over the map, with eve and the decoys marked
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor('#111')

    for r in range(env.height):
        for c in range(env.width):
            if env.grid[r, c] == 1:
                ax.plot(c, r, 's', color='#444', markersize=6)
            elif env.grid[r, c] == 2:
                ax.plot(c, r, 's', color='#2a6', markersize=7, alpha=0.7)

    for decoy in getattr(env, 'decoys', []):
        dr, dc = decoy['position']
        ax.plot(dc, dr, '^', color='#ff9a3c', markersize=11,
                markeredgecolor='black')

    unit_colors = ['#33cc33', '#3399ff', '#ff66cc', '#ffcc33', '#66ffff']
    for i, path in enumerate(ma_report['paths']):
        pr = [p[0] for p in path]
        pc = [p[1] for p in path]
        col = unit_colors[i % len(unit_colors)]
        ax.plot(pc, pr, '-', color=col, lw=1.4, alpha=0.8,
                label=f'Unit {i}')
        ax.plot(pc[0], pr[0], 's', color=col, markersize=8)
        ax.plot(pc[-1], pr[-1], 'o', color=col, markersize=9,
                markeredgecolor='white')

    ax.plot(env.eve_pos[1], env.eve_pos[0], '*', color='red',
            markersize=20, label='EVE', markeredgecolor='white')
    est = ma_report['final_estimate']
    ax.plot(est[1], est[0], 'x', color='yellow', markersize=12, mew=2,
            label='Shared estimate')

    title = (f"Coordinated fleet of {ma_report['n_agents']} — "
             f"{'FOUND by unit %s @ tick %s' % (ma_report['found_by_unit'], ma_report['found_step']) if ma_report['found'] else 'not found'}")
    ax.set_title(title, color='white')
    ax.set_xlabel('Column'); ax.set_ylabel('Row')
    ax.invert_yaxis()
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.15)

    import os
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a1a', edgecolor='none')
    plt.close()
    return output_path