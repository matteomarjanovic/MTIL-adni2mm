from torchsummary import summary
from st_model import _CNN

m = _CNN(10, 0.137)
summary(m, (1, 182, 218, 182))