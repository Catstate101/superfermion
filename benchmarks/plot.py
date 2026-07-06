
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    if not os.path.exists('benchmarks/results.csv'):
        print("results.csv not found")
        return
    
    df = pd.read_csv('benchmarks/results.csv')
    plt.figure(figsize=(15, 6))
    
    # Execution Latency (Forward)
    plt.subplot(1, 2, 1)
    for fw in df['fw'].unique():
        subset = df[df['fw'] == fw]
        plt.plot(subset['n'], subset['latency'], marker='o', label=fw)
    plt.yscale('log')
    plt.xlabel('Qubits')
    plt.ylabel('Latency (ms) - Log Scale')
    plt.title('Execution Speed Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gradient Latency (Sub-millisecond)
    plt.subplot(1, 2, 2)
    sf_grad = df[(df['fw'] == 'superfermion')]
    if not sf_grad.empty:
        plt.plot(sf_grad['n'], sf_grad['grad'], marker='s', label='Superfermion (Grad)', color='orange')
        plt.plot(sf_grad['n'], sf_grad['latency'], marker='o', label='Superfermion (Forward)', color='blue')
    plt.yscale('log')
    plt.xlabel('Qubits')
    plt.ylabel('Latency (ms)')
    plt.title('Superfermion Acceleration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('benchmarks/comparison_chart.png', dpi=300)
    print("Chart saved to benchmarks/comparison_chart.png")

if __name__ == "__main__":
    plot_results()
