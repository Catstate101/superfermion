"""
Multi-Tenancy Auth Layer — Security for the Superfermion Gateway.

Provides API Key validation, Role-based Access Control (RBAC), 
and Usage Quotas to prevent unauthorized access and DoS attacks.
"""

from __future__ import annotations

import secrets
from typing import Dict, Optional, Tuple
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_NAME = "X-SF-API-KEY"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Mock DB for API Keys and Quotas
# In production, this would be Redis/PostgreSQL
VAULT: Dict[str, Dict[str, Any]] = {
    "sf-master-key": {"user": "admin", "tier": "unlimited", "quota": 999999},
    "guest-key": {"user": "tester", "tier": "free", "quota": 10}
}

def authenticate(api_key: str = Security(api_key_header)) -> Dict[str, Any]:
    """Validates the API Key and returns user metadata."""
    if not api_key:
        raise HTTPException(status_code=401, detail="API Key missing. Access denied.")
        
    if api_key not in VAULT:
        raise HTTPException(status_code=403, detail="Invalid API Key.")
        
    user_data = VAULT[api_key]
    if user_data["quota"] <= 0:
         raise HTTPException(status_code=429, detail="Usage quota exceeded. Upgrade to Enterprise tier.")
         
    return user_data

def check_qubit_limit(n_qubits: int, tier: str):
    """Enforces qubit limits based on user tier."""
    LIMITS = {
        "free": 12,
        "pro": 28,
        "unlimited": 127
    }
    
    max_q = LIMITS.get(tier, 12)
    if n_qubits > max_q:
        raise HTTPException(
            status_code=403, 
            detail=f"Qubit count {n_qubits} exceeds {tier} tier limit of {max_q}."
        )
