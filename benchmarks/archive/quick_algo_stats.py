
import time
import superfermion as sf
from superfermion.backends.singularity import SingularityBackend

def get_algo_ms():
    # QAOA N=20
    c_qaoa = sf.Circuit(20)
    for i in range(20): c_qaoa.h(i)
    for _ in range(2):
        for i in range(20): c_qaoa.rzz(0.5, i, (i+1)%20)
        for i in range(20): c_qaoa.rx(0.3, i)
    
    SingularityBackend._topology_cache.clear()
    t0 = time.time(); sf.run(c_qaoa, backend="singularity"); t_cold = time.time()-t0
    t0 = time.time(); sf.run(c_qaoa, backend="singularity"); t_warm = time.time()-t0
    print(f"QAOA N=20: Cold={t_cold*1000:.1f}ms, Warm={t_warm*1000:.1f}ms")

    # QML N=20 (2 layers)
    c_qml = sf.Circuit(20)
    for _ in range(2):
        for i in range(20): c_qml.rx(0.1, i).ry(0.2, i).rz(0.3, i)
        for i in range(19): c_qml.cx(i, i+1)
        
    SingularityBackend._topology_cache.clear()
    t0 = time.time(); sf.run(c_qml, backend="singularity"); t_cold = time.time()-t0
    t0 = time.time(); sf.run(c_qml, backend="singularity"); t_warm = time.time()-t0
    print(f"QML N=20: Cold={t_cold*1000:.1f}ms, Warm={t_warm*1000:.1f}ms")

if __name__ == "__main__":
    get_algo_ms()
