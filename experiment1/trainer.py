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