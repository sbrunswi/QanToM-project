import torch.nn as nn
import torch as tr
import pennylane as qml


class CharNet(nn.Module):
    def __init__(self, num_past, num_input,device):
        super(CharNet, self).__init__()
        self.device = device
        self.conv = nn.Conv2d(num_input, 8, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTMCell(800, 800)
        self.avgpool = nn.AvgPool1d(8)
        self.fc1 = nn.Linear(100, 2)
        self.hidden_size = 800

    def init_hidden(self, batch_size):
        return  (tr.zeros(batch_size, 800, device=self.device),
                 tr.zeros(batch_size, 800, device=self.device))

    def forward(self, obs):
        # batch, num_past, step, channel , height, width
        b, num_past, num_step, c, h, w = obs.shape
        
        # Initialize e_char_sum as a tensor (not scalar 0) to handle num_past=0 case
        e_char_sum = tr.zeros((b, 2), device=self.device)
        
        for p in range(num_past):
            prev_h = self.init_hidden(b)
            obs_past = obs[:, p]
            obs_past = obs_past.permute(1, 0, 2, 3, 4)

            obs_past = obs_past.reshape(-1, c, h, w)
            x = self.conv(obs_past)
            x = self.relu(x)
            outs = []
            for step in range(num_step):
                out, prev_h = self.lstm(x.view(num_step, b, -1)[step], prev_h)
                outs.append(out)
            x = tr.stack(outs, dim=0)
            x = x.transpose(1, 0)
            x = self.avgpool(x)
            x = x.squeeze(1)
            x = self.fc1(x)
            e_char_sum += x

        return e_char_sum

class PredNet(nn.Module):
    def __init__(self, num_past, num_input, device):
        super(PredNet, self).__init__()
        self.device = device
        self.e_char = CharNet(num_past, num_input,device=device)
        self.conv1 = nn.Conv2d(8, 32, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(6)
        self.fc = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=-1)

    def init_hidden(self, batch_size):
        return self.e_char.init_hidden(batch_size)

    def forward(self, past_traj, obs):
        b, h, w, c = obs.shape
        obs = obs.permute(0, 3, 1, 2)
        _, num_past, _, _, _, _ = past_traj.shape
        if num_past == 0:
            e_char = tr.zeros((b, 2, h, w), device=self.device)
            e_char_2d = tr.zeros((b, 2), device=self.device)
        else:
            e_char_2d = self.e_char(past_traj)
            e_char = e_char_2d.unsqueeze(-1).unsqueeze(-1)
            e_char = e_char.repeat(1, 1, h, w)
        x_concat = tr.cat([e_char, obs], axis=1)


        x = self.relu(self.conv1(x_concat))
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)

        out = self.softmax(self.fc(x))

        return out, e_char_2d


class _QuantumCircuit:
    """Wrapper class to make quantum circuit picklable."""
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self._dev = None
        self._qnode = None
        self._build_circuit()
    
    def _build_circuit(self):
        """Build the quantum circuit and qnode."""
        self._dev = qml.device("default.qubit", wires=self.n_qubits)
        self._qnode = qml.QNode(self._circuit, self._dev, interface="torch")
    
    def _circuit(self, inputs, weights):
        """The actual quantum circuit."""
        # Angle embedding of input
        qml.AngleEmbedding(inputs, wires=range(self.n_qubits))
        
        # Variational layers
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        
        # Return expectation values
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
    
    def __call__(self, inputs, weights):
        """Make the class callable like a function."""
        return self._qnode(inputs, weights)
    
    def __getstate__(self):
        """Custom pickling: only save n_qubits, rebuild circuit on unpickle."""
        return {'n_qubits': self.n_qubits}
    
    def __setstate__(self, state):
        """Custom unpickling: restore n_qubits and rebuild circuit."""
        self.n_qubits = state['n_qubits']
        self._dev = None
        self._qnode = None
        self._build_circuit()


class QuantumStateEncoder(nn.Module):
    """Quantum state encoder using variational quantum circuits."""
    
    def __init__(self, input_dim, output_dim, n_qubits, n_layers=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Project input to qubit count
        self.input_projection = nn.Linear(input_dim, n_qubits)
        
        # Build quantum circuit (using class-based approach for picklability)
        quantum_circuit = _QuantumCircuit(n_qubits)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit._qnode, weight_shapes)
        # Store the circuit wrapper to keep it alive
        self._quantum_circuit = quantum_circuit
        
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


class PredNetQuantum(nn.Module):
    """Prediction network with quantum-enhanced final layer."""
    
    def __init__(self, num_past, num_input, device, n_qubits=4, n_layers=2, use_quantum=True):
        super(PredNetQuantum, self).__init__()
        self.device = device
        self.use_quantum = use_quantum
        
        # Classical character encoding (same as PredNet)
        self.e_char = CharNet(num_past, num_input, device=device)
        
        # Convolutional layers (same as PredNet)
        self.conv1 = nn.Conv2d(8, 32, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(6)
        
        # Quantum layer replaces the final linear layer (32 → 5)
        if self.use_quantum:
            # Input: 32 dims (from avgpool output)
            # Output: 5 dims (action probabilities)
            self.quantum_layer = QuantumStateEncoder(
                input_dim=32,
                output_dim=5,
                n_qubits=n_qubits,
                n_layers=n_layers
            )
        else:
            # Fallback to classical linear layer
            self.fc = nn.Linear(32, 5)
        
        self.softmax = nn.Softmax(dim=-1)

    def init_hidden(self, batch_size):
        return self.e_char.init_hidden(batch_size)

    def forward(self, past_traj, obs):
        b, h, w, c = obs.shape
        obs = obs.permute(0, 3, 1, 2)
        _, num_past, _, _, _, _ = past_traj.shape
        
        if num_past == 0:
            e_char = tr.zeros((b, 2, h, w), device=self.device)
            e_char_2d = tr.zeros((b, 2), device=self.device)
        else:
            e_char_2d = self.e_char(past_traj)
            e_char = e_char_2d.unsqueeze(-1).unsqueeze(-1)
            e_char = e_char.repeat(1, 1, h, w)
        
        x_concat = tr.cat([e_char, obs], axis=1)
        
        x = self.relu(self.conv1(x_concat))
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)  # Shape: (b, 32)
        
        # Quantum layer replaces the final linear layer
        if self.use_quantum:
            out = self.quantum_layer(x)  # Quantum: (b, 32) -> (b, 5)
        else:
            out = self.fc(x)  # Classical: (b, 32) -> (b, 5)
        
        out = self.softmax(out)
        
        return out, e_char_2d