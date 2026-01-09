"""
Enhanced Theory of Mind observer models allowing independent variation of
state encoders.

- State encoders: Classical | Quantum | Hybrid | VAE

This module is used by experiments that compare state-encoder types in isolation.
The `EnhancedToMObserver` fuses character, mental, and encoded_state into a
policy head that predicts the next action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if PennyLane is available
try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except Exception as e:
    qml = None
    _HAS_PENNYLANE = False

class ClassicalStateEncoder(nn.Module):
    """Classical state encoder using neural networks."""
    
    def __init__(self, input_dim: int = 17, output_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)

class VAEStateEncoder(nn.Module):
    """Variational Autoencoder-based state encoder with parameter count matched
    to quantum/hybrid encoders.

    Encodes input to a latent vector via reparameterization and returns the
    latent mean as the state representation. A tiny decoder is included to keep
    overall parameterization comparable, but its output is not used by the
    observer forward.
    """
    def __init__(self, input_dim: int = 17, output_dim: int = 32,
                 hidden_dim: int = 24, decoder_hidden_dim: int = 16):
        super().__init__()
        # Encoder to mean and logvar (single compact hidden layer)
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, output_dim)
        self.enc_logvar = nn.Linear(hidden_dim, output_dim)
        # Tiny decoder to roughly match parameter count to quantum/hybrid
        self.dec_fc1 = nn.Linear(output_dim, decoder_hidden_dim)
        self.dec_out = nn.Linear(decoder_hidden_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor):
        h = F.relu(self.enc_fc1(x))
        mu = torch.tanh(self.enc_mu(h))
        logvar = self.enc_logvar(h)
        return mu, logvar

    def decode(self, z: torch.Tensor):
        h = F.relu(self.dec_fc1(z))
        recon = self.dec_out(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        # Use mean as deterministic representation during inference/training for fairness
        z = mu
        return z

class QuantumStateEncoder(nn.Module):
    """Quantum state encoder using variational quantum circuits."""
    
    def __init__(self, input_dim: int = 17, output_dim: int = 32, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Quantum state encoder requires pennylane to be installed."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Project input to qubit count
        self.input_projection = nn.Linear(input_dim, n_qubits)
        
        # Build quantum circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_state_circuit(inputs, weights):
            # Angle embedding of input
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_state_circuit, weight_shapes)
        
        # Post-processing to output dimension
        self.post_process = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Project input to qubit count
        projected_input = self.input_projection(x)
        
        # Process through quantum circuit
        quantum_output = self.quantum_layer(projected_input)
        
        # Post-process to final output
        return self.post_process(quantum_output)

class HybridStateEncoder(nn.Module):
    """Hybrid state encoder combining classical and quantum components."""
    
    def __init__(self, input_dim: int = 17, output_dim: int = 32, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        assert _HAS_PENNYLANE, "Hybrid state encoder requires pennylane to be installed."
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Classical component
        self.classical_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, output_dim // 2),
            nn.Tanh()
        )
        
        # Quantum component
        self.input_projection = nn.Linear(input_dim, n_qubits)
        
        # Build quantum circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def quantum_state_circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_state_circuit, weight_shapes)
        
        # Quantum post-processing
        self.quantum_post_process = nn.Sequential(
            nn.Linear(n_qubits, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim // 2),
            nn.Tanh()
        )
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Classical encoding
        classical_output = self.classical_encoder(x)
        
        # Quantum encoding
        projected_input = self.input_projection(x)
        quantum_output = self.quantum_layer(projected_input)
        quantum_output = self.quantum_post_process(quantum_output)
        
        # Fuse outputs
        combined = torch.cat([classical_output, quantum_output], dim=-1)
        return self.fusion_layer(combined)

## Belief-state modules removed

class EnhancedToMObserver(nn.Module):
    """
    Enhanced ToM-style observer with pluggable state and belief components.

    Inputs
    ------
    - char:   Character summary embedding input (B, char_dim)
    - mental: Mental window embedding input (B, mental_dim)
    - state:  Raw state features for state encoder (B, state_dim)

    Configuration
    -------------
    - state_type:  'classical' | 'quantum' | 'hybrid' | 'vae'
    - n_qubits:    qubits used by quantum/hybrid modules

    Flow
    ----
    state --(state_encoder)--> encoded_state
    [char, mental, encoded_state] --concat--> policy head --> logits
    """
    
    def __init__(self, char_dim: int = 22, mental_dim: int = 17, state_dim: int = 17,
                 state_type: str = "classical", belief_type: str = "classical",
                 n_qubits: int = 8, device: str = "cpu"):
        super().__init__()
        
        assert state_type in {"classical", "quantum", "hybrid", "vae"}
        
        self.state_type = state_type
        # belief_type accepted for backward-compatibility; ignored
        self.device = device
        
        # Character encoder
        self.char_enc = nn.Sequential(
            nn.Linear(char_dim, char_dim), nn.ReLU(),
            nn.Linear(char_dim, char_dim), nn.ReLU(),
        )
        
        # Mental encoder
        self.mental_enc = nn.Sequential(
            nn.Linear(mental_dim, mental_dim), nn.ReLU(),
            nn.Linear(mental_dim, mental_dim), nn.ReLU(),
        )
        
        # State encoder
        if state_type == "classical":
            self.state_encoder = ClassicalStateEncoder(state_dim, 32)
        elif state_type == "quantum":
            self.state_encoder = QuantumStateEncoder(state_dim, 32, n_qubits)
        elif state_type == "hybrid":
            self.state_encoder = HybridStateEncoder(state_dim, 32, n_qubits // 2)
        else:  # vae
            # Use compact VAE to roughly match quantum/hybrid parameter counts
            self.state_encoder = VAEStateEncoder(state_dim, 32, hidden_dim=24, decoder_hidden_dim=16)
        
        # Policy head: combines character, mental state, and encoded state
        fused_dim = char_dim + mental_dim + 32  # char + mental + encoded_state
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 5)
        )
    
    def forward(self, char, mental, state):
        # Encode character past behavior and recent mental context
        c = self.char_enc(char)
        m = self.mental_enc(mental)

        # Encode current state
        encoded_state = self.state_encoder(state)

        # Fuse and predict next action logits
        x = torch.cat([c, m, encoded_state], dim=-1)

        
        logits = self.head(x)
        return logits
    
    def get_state_representation(self, state):
        """Return state encoder output for analysis/visualization."""
        return self.state_encoder(state)
    
    def get_belief_representation(self, state):
        """Deprecated: returns the state encoder embedding for compatibility."""
        return self.state_encoder(state)

def create_enhanced_tom_observer(state_type: str, belief_type: str = "classical", **kwargs):
    """Factory function to create enhanced ToM observers."""
    return EnhancedToMObserver(state_type=state_type, belief_type=belief_type, **kwargs)
