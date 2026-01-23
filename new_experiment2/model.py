import torch.nn as nn
import torch as tr
import torch.nn.functional as F
from tqdm import tqdm
import pennylane as qml

def cross_entropy_with_soft_label(pred, targ):
    return -(targ * pred.log()).sum(dim=-1).mean()

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv_function(x)
        return x

class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * ResNetBlock.expansion, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * ResNetBlock.expansion),
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != ResNetBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResNetBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResNetBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class softmax_SR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sr = F.softmax(x.reshape(x.size(0), x.size(1), -1), dim=2)
        sr = sr.transpose(1, 2)
        return sr

class CharNet(nn.Module):
    def __init__(self, num_past, num_input, num_step, num_exp=1,device=None):
        super(CharNet, self).__init__()
        self.device = device
        self.num_exp = num_exp
        self.conv1 = ResNetBlock(num_input, 4, 1)
        self.conv2 = ResNetBlock(4, 8, 1)
        self.conv3 = ResNetBlock(8, 16, 1)
        self.conv4 = ResNetBlock(16, 32, 1)
        self.conv5 = ResNetBlock(32, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, 64)
        self.avgpool = nn.AvgPool2d(11)
        self.fc64_2 = nn.Linear(num_step * 64, 2)
        self.fc64_8 = nn.Linear(num_step * 64, 8)
        self.fc32_2 = nn.Linear(32, 2)
        self.fc32_8 = nn.Linear(32, 8)
        self.hidden_size = 64

    def init_hidden(self, batch_size):
        return (tr.zeros(1, batch_size, 64, device=self.device),
                tr.zeros(1, batch_size, 64, device=self.device))

    def forward(self, obs):
        # batch, num_past, num_step, height, width, channel
        obs = obs.permute(0, 1, 2, 5, 3, 4)
        b, num_past, num_step, c, h, w = obs.shape
        past_e_char = []
        for p in range(num_past):
            prev_h = self.init_hidden(b)
            obs_past = obs[:, p] #batch(0), num_step(1), channel(2), height(3), width(4)
            obs_past = obs_past.permute(1, 0, 2, 3, 4)
            obs_past = obs_past.reshape(-1, c, h, w)

            x = self.conv1(obs_past)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.relu(x)
            x = self.bn(x)
            x = self.avgpool(x)

            if self.num_exp == 2:
                x = x.view(num_step, b, -1)
                x = x.transpose(1, 0)
                x = self.fc32_8(x)  ## batch, output
                past_e_char.append(x)  # Accumulate for all past episodes
            else:
                outs, hidden = self.lstm(x.view(num_step, b, -1), prev_h)
                outs = outs.transpose(0, 1).reshape(b, -1) ## batch, step * output
                e_char_sum = self.fc64_8(outs) ## batch, output
                past_e_char.append(e_char_sum)

        # Sum all past episode embeddings
        if len(past_e_char) > 0:
            past_e_char = tr.stack(past_e_char, dim=0)
            final_e_char = tr.sum(past_e_char, dim=0)  # Sum over past episodes: (num_past, batch, 8) -> (batch, 8)
        else:
            # Fallback if no past episodes (shouldn't happen, but safe)
            final_e_char = tr.zeros((b, 8), device=self.device)

        return final_e_char

class PredNet(nn.Module):
    def __init__(self, num_past, num_input, num_agent, num_step, device, num_exp=2):
        super(PredNet, self).__init__()

        self.e_char = CharNet(num_past, num_input, num_exp=num_exp, num_step=num_step,device=device)
        self.conv1 = ResNetBlock(14, 8, 1)
        self.conv2 = ResNetBlock(8, 16, 1)
        self.conv3 = ResNetBlock(16, 16, 1)
        self.conv4 = ResNetBlock(16, 32, 1)
        self.conv5 = ResNetBlock(32, 32, 1)
        self.normal_conv1 = ConvBlock(14, 8, 1)
        self.normal_conv2 = ConvBlock(8, 16, 1)
        self.normal_conv3 = ConvBlock(16, 16, 1)
        self.normal_conv4 = ConvBlock(16, 32, 1)
        self.normal_conv5 = ConvBlock(32, 32, 1)
        self.avgpool = nn.AvgPool2d(11)
        self.bn = nn.BatchNorm2d(32)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()
        self.action_fc = nn.Linear(32, 5)
        self.comsumption_fc = nn.Linear(32, 4)
        self.device = device
        self.softmax = nn.Softmax()
        self.num_agent = num_agent

        self.action_head = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(11),
            nn.Flatten(),
            nn.Linear(32,5),
            nn.LogSoftmax(dim=-1)
        )

        self.consumption_head = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(11),
            nn.Flatten(),
            nn.Linear(32, 5),
            nn.LogSoftmax(dim=-1)
        )

        self.representation_head = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, 1),
            softmax_SR() # each channel is for gamma=0.5, 0.9, 0.99
        )

    def init_hidden(self, batch_size):
        return self.e_char.init_hidden(batch_size)

    def forward(self, past_traj, obs):
        b, h, w, c = obs.shape
        obs = obs.permute(0, 3, 1, 2)
        _, _, s, _, _, _ = past_traj.shape
        if s == 0:
            # When no past trajectories, create zero e_char with correct dimensions
            # CharNet returns (batch, 8) for experiment 2, so we need (batch, 8, h, w)
            e_char_2d = tr.zeros((b, 8), device=self.device)
            e_char = tr.zeros((b, 8, h, w), device=self.device)
        else:
            e_char_2d = self.e_char(past_traj)
            # e_char_2d should have shape (batch, 8)
            # Ensure it has the correct shape
            if len(e_char_2d.shape) == 2:
                # (batch, 8) -> (batch, 8, 1, 1) -> (batch, 8, h, w)
                e_char = e_char_2d.view(b, -1, 1, 1).expand(b, -1, h, w)
            else:
                # If shape is unexpected, reshape to (batch, 8)
                e_char_2d = e_char_2d.view(b, -1)
                if e_char_2d.shape[1] != 8:
                    # If not 8 dimensions, take first 8 or pad/truncate
                    if e_char_2d.shape[1] > 8:
                        e_char_2d = e_char_2d[:, :8]
                    else:
                        padding = tr.zeros((b, 8 - e_char_2d.shape[1]), device=self.device)
                        e_char_2d = tr.cat([e_char_2d, padding], dim=1)
                e_char = e_char_2d.view(b, 8, 1, 1).expand(b, 8, h, w)
        x_concat = tr.cat([e_char, obs], axis=1)

        x = self.conv1(x_concat)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn(x)

        action = self.action_head(x)
        consumption = self.consumption_head(x)
        sr = self.representation_head(x)

        return action, consumption, sr, e_char_2d


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
    """Prediction network with quantum-enhanced heads (action and consumption)."""
    
    def __init__(self, num_past, num_input, num_agent, num_step, device, num_exp=2, 
                 n_qubits=4, n_layers=2, use_quantum=True):
        super(PredNetQuantum, self).__init__()
        
        self.device = device
        self.use_quantum = use_quantum
        self.num_agent = num_agent
        
        # Character network (same as PredNet)
        self.e_char = CharNet(num_past, num_input, num_exp=num_exp, num_step=num_step, device=device)
        
        # Convolutional layers (same as PredNet)
        self.conv1 = ResNetBlock(14, 8, 1)
        self.conv2 = ResNetBlock(8, 16, 1)
        self.conv3 = ResNetBlock(16, 16, 1)
        self.conv4 = ResNetBlock(16, 32, 1)
        self.conv5 = ResNetBlock(32, 32, 1)
        self.avgpool = nn.AvgPool2d(11)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        
        # Quantum-enhanced or classical heads
        if self.use_quantum:
            # Quantum-enhanced action head
            self.action_preprocess = nn.Sequential(
                nn.Conv2d(32, 32, 1, 1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(11),
                nn.Flatten()
            )
            self.action_quantum = QuantumStateEncoder(
                input_dim=32,
                output_dim=5,
                n_qubits=n_qubits,
                n_layers=n_layers
            )
            self.action_logsoftmax = nn.LogSoftmax(dim=-1)
            
            # Quantum-enhanced consumption head
            self.consumption_preprocess = nn.Sequential(
                nn.Conv2d(32, 32, 1, 1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(11),
                nn.Flatten()
            )
            self.consumption_quantum = QuantumStateEncoder(
                input_dim=32,
                output_dim=5,
                n_qubits=n_qubits,
                n_layers=n_layers
            )
            self.consumption_logsoftmax = nn.LogSoftmax(dim=-1)
        else:
            # Classical heads (fallback)
            self.action_head = nn.Sequential(
                nn.Conv2d(32, 32, 1, 1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(11),
                nn.Flatten(),
                nn.Linear(32, 5),
                nn.LogSoftmax(dim=-1)
            )
            self.consumption_head = nn.Sequential(
                nn.Conv2d(32, 32, 1, 1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(11),
                nn.Flatten(),
                nn.Linear(32, 5),
                nn.LogSoftmax(dim=-1)
            )
        
        # Representation head (always classical, outputs spatial maps)
        self.representation_head = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, 1),
            softmax_SR()  # each channel is for gamma=0.5, 0.9, 0.99
        )
    
    def init_hidden(self, batch_size):
        return self.e_char.init_hidden(batch_size)
    
    def forward(self, past_traj, obs):
        b, h, w, c = obs.shape
        obs = obs.permute(0, 3, 1, 2)
        _, _, s, _, _, _ = past_traj.shape
        
        if s == 0:
            # When no past trajectories, create zero e_char with correct dimensions
            e_char_2d = tr.zeros((b, 8), device=self.device)
            e_char = tr.zeros((b, 8, h, w), device=self.device)
        else:
            e_char_2d = self.e_char(past_traj)
            # Ensure e_char_2d has correct shape (batch, 8)
            if len(e_char_2d.shape) == 2:
                e_char = e_char_2d.view(b, -1, 1, 1).expand(b, -1, h, w)
            else:
                e_char_2d = e_char_2d.view(b, -1)
                if e_char_2d.shape[1] != 8:
                    if e_char_2d.shape[1] > 8:
                        e_char_2d = e_char_2d[:, :8]
                    else:
                        padding = tr.zeros((b, 8 - e_char_2d.shape[1]), device=self.device)
                        e_char_2d = tr.cat([e_char_2d, padding], dim=1)
                e_char = e_char_2d.view(b, 8, 1, 1).expand(b, 8, h, w)
        
        x_concat = tr.cat([e_char, obs], axis=1)
        
        x = self.conv1(x_concat)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn(x)
        
        # Action and consumption predictions
        if self.use_quantum:
            # Quantum-enhanced action head
            action_features = self.action_preprocess(x)  # (batch, 32)
            action = self.action_quantum(action_features)  # (batch, 5)
            action = self.action_logsoftmax(action)
            
            # Quantum-enhanced consumption head
            consumption_features = self.consumption_preprocess(x)  # (batch, 32)
            consumption = self.consumption_quantum(consumption_features)  # (batch, 5)
            consumption = self.consumption_logsoftmax(consumption)
        else:
            # Classical heads
            action = self.action_head(x)
            consumption = self.consumption_head(x)
        
        # Successor representation (always classical)
        sr = self.representation_head(x)
        
        return action, consumption, sr, e_char_2d
