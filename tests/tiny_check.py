import superfermion as sf
c = sf.Circuit(10).h(0)
for i in range(9): c.cx(i, i+1)
res = sf.run(c, backend="singularity", shots=1000)
print(f"Singularity OK: {res.counts}")
res = sf.run(c, backend="rust", shots=1000)
print(f"Rust OK: {res.counts}")
