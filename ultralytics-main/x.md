解决方法：找到ultralytics/engine/trainer.py文件，修改backward：

# Backward
torch.use_deterministic_algorithms(False) # 禁用确定性算法模式

self.scaler.scale(self.loss).backward()