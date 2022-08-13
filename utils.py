# 라이브러리 불러오기 
import torch
import torch.nn as nn
import numpy as np

# 은닉층 정의
# 은닉층 하나를 LinearBlock으로 만들어 설계 
class LinearBlock(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(LinearBlock, self).__init__()
        self.layer = nn.utils.weight_norm(nn.Linear(in_nodes, out_nodes), dim = 0) # nn.Linear를 통해 층을 하나 만들고 nn.utils.weight_norm을 이용, 정규화된 가중치를 사용하도록 함. 

    def forward(self, x):
        x = self.layer(x)
        x = x * torch.sigmoid(x) # SiLU # 은닉층에서는 활성화 함수임 SiLU 함수 x.sigmoid(x) 적용 
        return x

# 모델 정의 
class PINN(nn.Module):

    def __init__(self, data, layer_list): # 모델 구축에 관한 함수 
        super(PINN, self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_layer = nn.utils.weight_norm(nn.Linear(layer_list[0], layer_list[1]), dim = 0) # 입력층 정의 
        self.hidden_layers = self._make_layer(layer_list[1:-1]) # 은닉층 정의 
        self.output_layer = nn.utils.weight_norm(nn.Linear(layer_list[-2], layer_list[-1]), dim = 0) # 출력층 정의 
        self.data = data # 데이터를 받아 평균 & 표준편차 계산 
        self.mean = self.data.mean(dim=0).to(device)
        self.sig = torch.sqrt(self.data.var(dim=0)).to(device)

    def _make_layer(self, layer_list): # 은닉층의 노드 정보가 담 리스트 layer_list를 받음.
        layers = [] # 은닉층을 쌓을 빈 리스트 생성 
        for i in range(len(layer_list) - 1): # 미리 정의한 LinearBlock을 불러와 차례대로 쌓아 은닉층을 nn.Sequential로 반환. 
            block = LinearBlock(layer_list[i], layer_list[i + 1])
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x): # 실제 연산이 일어남 
        x = (x - self.mean) / self.sig # 미리 구한 데이터의 평균 & 표준편차 이용, 입력 x, y, t 데이터 정규화 
        x = self.input_layer(x) # 입력층 계산
        x = x * torch.sigmoid(x)
        x = self.hidden_layers(x) # 그 다음 은닉층 계산 
        x = self.output_layer(x) # 출력층 계산 => c, u, v, p 출력 
        return x

# 초기 모델 변수 설정
# nn.Linear에 대해 torch.nn.init.xavier_normal을 이용, 초기 모델 변수를 정규화한 값으로 설정 
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)

# 모델 불러오기 
def pinn(data, layer_list): # 모델의 각 층의 노드 수를 받아 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PINN(data, layer_list).to(device) # 모델을 불러온 뒤 
    model.apply(weights_init) # 초기 모델 변수 설정한 후 
    print("Operation mode: ", device)
    return model # 모델 반환 

# 자동 미분 함수 구현 
def fwd_gradients(obj, x):
    dummy = torch.ones_like(obj)
    derivative = torch.autograd.grad(obj, x, dummy, create_graph= True)[0] # torch.autograd.grad를 이용, 미분값을 구함.
    # ex. u의 x에 대한 미분 계산 시 코드 내 obj에 미분하고자 하는 함수 u 입력 & x에는 미분의 기준이 되는 변수 x를 입력.
    # 출력할 미분값을 위한 출력 벡터와 크기가 같은 벡터 dummy를 넣어주고 create_graph 활성화, 미분 그래프 생
    return derivative

# 나비에-스톡스 방정식 계산 
def Navier_Stokes_2D(c, u, v, p, txy, Pec, Rey): # 예측값 c, u, v, p & t, x, y좌표 데이터인 txy 불러옴. (Pec=100, Rey=100)
    c_txy = fwd_gradients(c, txy) # 각 변수 c, u, v, p를 t, x, y에 대해 미분, 이때 t, x, y에 대한 미분값이 반환됨. 
    u_txy = fwd_gradients(u, txy)
    v_txy = fwd_gradients(v, txy)
    p_txy = fwd_gradients(p, txy)

    c_t = c_txy[:, 0:1] # 다음 계산된 t, x, y에 대한 미분을 각각 나누어 정의, 0, 1, 2열은 t, x, y에 대한 미분임. 2차원 유지를 위하여 [:,0](X), [:,0:1](O) 
    c_x = c_txy[:, 1:2]
    c_y = c_txy[:, 2:3]
    u_t = u_txy[:, 0:1]
    u_x = u_txy[:, 1:2]
    u_y = u_txy[:, 2:3]
    v_t = v_txy[:, 0:1]
    v_x = v_txy[:, 1:2]
    v_y = v_txy[:, 2:3]
    p_x = p_txy[:, 1:2]
    p_y = p_txy[:, 2:3]

    c_xx = fwd_gradients(c_x, txy)[:, 1:2] # 두 번 미분한 값을 구하기 위해 fwd_gradients 미분한 것을 또 미분! 
    c_yy = fwd_gradients(c_y, txy)[:, 2:3]
    u_xx = fwd_gradients(u_x, txy)[:, 1:2]
    u_yy = fwd_gradients(u_y, txy)[:, 2:3]
    v_xx = fwd_gradients(v_x, txy)[:, 1:2]
    v_yy = fwd_gradients(v_y, txy)[:, 2:3]

    e1 = c_t + (u * c_x + v * c_y) - (1.0 / Pec) * (c_xx + c_yy) # 마지막으로 우리가 원하는 e1~e4를 만들어 반환 
    e2 = u_t + (u * u_x + v * u_y) + p_x - (1.0 / Rey) * (u_xx + u_yy)
    e3 = v_t + (u * v_x + v * v_y) + p_y - (1.0 / Rey) * (v_xx + v_yy)
    e4 = u_x + v_y

    return e1, e2, e3, e4

def test_data(T_star, X_star, Y_star, C_star, U_star, V_star, P_star):
    snap = np.random.randint(0, T_star.shape[1]) # 모든 지점에 대해 평가하면 시간 오래 걸려서 특정 시간에 대한 데이터 추출해 반환. 
    t_star = T_star[:, snap:snap+1]
    x_star = X_star[:, snap:snap+1]
    y_star = Y_star[:, snap:snap+1]
    c_star = C_star[:, snap:snap+1]
    u_star = U_star[:, snap:snap+1]
    v_star = V_star[:, snap:snap+1]
    p_star = P_star[:, snap:snap+1]

    variables_star = torch.FloatTensor(np.concatenate((t_star, x_star, y_star), 1))  # N x 3 # 입력 데이터 생성 
    target_star = torch.FloatTensor(np.concatenate((c_star, u_star, v_star, p_star), 1))  # N x 4 # 예측값과 비교를 위해 실제값 만들어 반환 

    return variables_star, target_star

# 상대오차 정의
# 실제값 target과 예측값 pred를 받아 L2 상대오차를 통해 성능 확인 
def relative_error(pred, target):
    return torch.sqrt(torch.mean((pred - target)**2)/torch.mean((target - torch.mean(target))**2))
