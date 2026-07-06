
import superfermion as sf
import numpy as np
import time

def test_ghz_match():
    n = 4
    c = sf.Circuit(n).h(0)
    for i in range(n-1):
        c.cx(i, i+1)
    
    shots = 4096
    res = sf.run(c, backend="singularity", shots=shots, seed=42)
    print(f"SF GHZ Counts: {res.counts}")
    
    # Expected: '0000' and '1111' only
    expected = {"0000": 2048, "1111": 2048} # approx
    counts = res.counts
    match = (counts.get("0000", 0) > 1800 and counts.get("1111", 0) > 1800)
    print(f"GHZ Accuracy Match (heuristic): {match}")
    if not match:
        print(f"TOTAL COUNTS: {sum(counts.values())}")
        print(f"ALL KEYS: {list(counts.keys())}")

if __name__ == "__main__":
    test_ghz_match()
