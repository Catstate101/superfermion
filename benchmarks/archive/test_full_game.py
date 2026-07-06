
import json
import requests
import superfermion as sf
from superfermion.circuit import Circuit
from superfermion.simulator import simulate_statevector

# ========================================================================
# 1. THE SUPERFERMION GAME CLIENT
# ========================================================================
class GameClient:
    def __init__(self, base_url="https://demo-entanglement-distillation-qfhvrahfcq-uc.a.run.app", api_token=None):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.player_id = None
        self._cached_graph = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.api_token: h["Authorization"] = f"Bearer {self.api_token}"
        return h

    def register(self, player_id, name, location="remote"):
        url = f"{self.base_url}/v1/register"
        resp = requests.post(url, json={"player_id": player_id, "name": name, "location": location})
        data = resp.json()
        if data.get("ok"):
            self.player_id = player_id
            self.api_token = data["data"].get("api_token")
        return data

    def select_starting_node(self, node_id):
        url = f"{self.base_url}/v1/select_starting_node"
        payload = {"player_id": self.player_id, "node_id": node_id}
        resp = requests.post(url, json=payload, headers=self._headers())
        return resp.json()

    def get_status(self):
        url = f"{self.base_url}/v1/status/{self.player_id}"
        resp = requests.get(url, headers=self._headers())
        return resp.json().get("data", {})

    def get_claimable_edges(self):
        status = self.get_status()
        owned = set(status.get('owned_nodes', []))
        graph_url = f"{self.base_url}/v1/graph"
        graph = requests.get(graph_url).json().get("data", {})
        claimable = []
        for edge in graph.get('edges', []):
            n1, n2 = edge['edge_id']
            if (n1 in owned) != (n2 in owned):
                claimable.append(edge)
        return claimable

    def claim_edge(self, edge, circuit: Circuit, flag_bit: int):
        # Superfermion QASM 3.0 Export
        qasm = circuit.to_qasm3()
        # Ensure XOR logic is present for the game server
        if "c[2] = c[0] ^ c[1];" not in qasm:
             lines = qasm.splitlines()
             if lines and lines[-1] == "": lines.pop()
             lines.append("c[2] = c[0] ^ c[1];")
             qasm = "\n".join(lines)
             
        payload = {
            "player_id": self.player_id,
            "edge": list(edge),
            "num_bell_pairs": 2,
            "circuit_qasm": qasm,
            "flag_bit": flag_bit,
        }
        resp = requests.post(f"{self.base_url}/v1/claim_edge", json=payload, headers=self._headers())
        return resp.json()

    def get_leaderboard(self):
        resp = requests.get(f"{self.base_url}/v1/leaderboard")
        return resp.json().get("data", {}).get("leaderboard", [])

# ========================================================================
# 2. RUNNING THE GAME TEST
# ========================================================================
import time
def run_test():
    client = GameClient()
    PLAYER_ID = f"GiselleRocio_SF_{int(time.time())}"
    
    print("--- STEP 1: REGISTRATION ---")
    reg = client.register(PLAYER_ID, PLAYER_ID)
    if reg.get("ok"):
        print(f"Registered! Token: {client.api_token[:15]}...")
        candidates = reg["data"].get("starting_candidates", [])
        for c in candidates:
            print(f"  - {c['node_id']}: {c['utility_qubits']} qubits, +{c['bonus_bell_pairs']} bonus")
            
        print("\n--- STEP 2: SELECT STARTING NODE ---")
        start_node = candidates[0]['node_id']
        sel = client.select_starting_node(start_node)
        print(f"Result: {sel.get('ok')}, Node: {start_node}")
    else:
        print(f"Registration failed: {reg.get('error')}")
        return

    print("\n--- STEP 3: STATUS CHECK ---")
    status = client.get_status()
    print(f"Score: {status.get('score')} | Budget: {status.get('budget')}")
    
    print("\n--- STEP 4: CLAIMABLE EDGES ---")
    claimable = client.get_claimable_edges()
    for e in claimable[:3]:
        print(f"  - {e['edge_id']} | Diff: {e['difficulty_rating']} | Thr: {e['base_threshold']}")

    print("\n--- STEP 5: DESIGN & SUBMIT PROTOCOL (SUPERFERMION) ---")
    # Build BBPSSW with SF
    c = Circuit(4, 3)
    c.cnot(1, 0).cnot(2, 3)
    c.measure(0, 0).measure(3, 1)
    
    target_edge = claimable[0]['edge_id']
    print(f"Attempting claim on {target_edge}...")
    result = client.claim_edge(target_edge, c, flag_bit=2)
    
    data = result.get("data", {})
    print(f"Success: {data.get('success')} | Fidelity: {data.get('fidelity', 0):.4f}")

    print("\n--- STEP 6: LEADERBOARD ---")
    lb = client.get_leaderboard()
    for i, p in enumerate(lb[:5]):
        print(f"{i+1}. {p['player_id']:20} | Score: {p['score']}")

if __name__ == "__main__":
    run_test()
