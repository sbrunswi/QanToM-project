import torch.nn as nn
import torch as tr


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
        e_char_sum = 0
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
        _, _, s, _, _, _ = past_traj.shape
        if s == 0:
            e_char = tr.zeros((b, 2, h, w), device=self.device)
            # BUG: e_char_2d is not defined when s == 0, but returned on line 81 - will cause NameError
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
    
class trainer(self,model,device='cpu'):
     self.model = model.to(device)
     self.device = device
     self.criterion = nn.KLDivLoss(reduction="batchmean")

    def train_epoch(self, data_loader, optimizer):
            self.model.train()
            
            tot_acc = 0
            tot_loss = 0

            for past_traj, curr_state, target in data_loader:
                past_traj = past_traj.float().to(self.device)
                curr_state = curr_state.float().to(self.device)
                target = target.float().to(self.device)

                optimizer.zero_grad()

                pred, _ = self.model(past_traj, curr_state)
                pred = pred.clamp(min=1e-8)

                loss = self.criterion(pred.log(), target)
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                tot_acc += (pred.argmax(-1) == target.argmax(-1)).sum().item()

            return {
                "action_loss": tot_loss / len(data_loader),
                "action_acc": tot_acc / len(data_loader.dataset),
            }

    def evaluate(self, data_loader):
        self.model.eval()
        tot_loss, tot_acc = 0, 0

        with tr.no_grad():
            for past_traj, curr_state, target in data_loader:
                past_traj = past_traj.float().to(self.device)
                curr_state = curr_state.float().to(self.device)
                target = target.float().to(self.device)

                pred, _ = self.model(past_traj, curr_state)
                pred = pred.clamp(min=1e-8)

                loss = self.criterion(pred.log(), target)
                tot_loss += loss.item()
                tot_acc += (pred.argmax(-1) == target.argmax(-1)).sum().item()

        return {
            "action_loss": tot_loss / len(data_loader),
            "action_acc": tot_acc / len(data_loader.dataset),
        }

