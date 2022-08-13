# 라이브러리 불러오기
import scipy.io # mat 파일을 넘파이 배열로 변환하는 함수 제공 
import numpy as np
import torch

# 데이터 처리 함수 정의 
def load_data(path, num_samples): # 데이터 경로, 뽑을 샘플 수 입력 
    data = scipy.io.loadmat(path) # mat 파일을 넘파이 배열이 있는 딕셔너리 타입으로 불러옴. 

    t_star = data['t_star']  # T x 1 # 각 키를 통해 t, x, y, U, V, P, C 데이터 불러오기 
    x_star = data['x_star']  # N x 1
    y_star = data['y_star']  # N x 1

    T = t_star.shape[0]
    N = x_star.shape[0]

    U_star = data['U_star']  # N x T
    V_star = data['V_star']  # N x T
    P_star = data['P_star']  # N x T
    C_star = data['C_star']  # N x T
    T_star = np.tile(t_star, (1, N)).T  # N x T # x, y는 좌표값 => 모든 타임 스탭에 동일한 값으로 존재 => title을 통해 반복 배열 생성. 그에 따른 시간 t도 동일 크기로 배열 생성. 
    X_star = np.tile(x_star, (1, T))  # N x T
    Y_star = np.tile(y_star, (1, T))  # N x T
    # 다음으로 우리 모델은 MLP에 x, y, t값이 들어감, 데이터의 크기 = (배치 사이즈X3)이 되어야 함.

    # For Training
    idx_t = np.concatenate([np.array([0]), np.random.choice(T - 2, T - 2, replace=False) + 1, np.array([T - 1])]) # 이를 위해 각 변수를 일렬로 펴서 (배치 사이즈X1)로 만든 뒤 세 변수를 합쳐 (배치 사이즈X3) 크기의 데이터 확보.
    # idx_t는 시간에 따라 데이터를 섞어주는 인덱스, idx_x는 각 좌표를 무작위로 샘플 수만큼 뽑는 인덱스
    idx_x = np.random.choice(N, num_samples, replace=False)
    t_data = T_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    x_data = X_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1 # 각 변수의 일렬 데이터를 얻음.
    y_data = Y_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    c_data = C_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1

    c_tensor_data = torch.FloatTensor(c_data)

    idx_t = np.concatenate([np.array([0]), np.random.choice(T - 2, T - 2, replace=False) + 1, np.array([T - 1])]) # 차후 별도로 모델 활성화를 위하여 바로 앞에서 했던 작업을 한 번 더! 
    idx_x = np.random.choice(N, num_samples, replace=False)
    t_eqns = T_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    x_eqns = X_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1
    y_eqns = Y_star[:, idx_t][idx_x, :].flatten()[:, None] # ST x 1

    variables = torch.FloatTensor(np.concatenate((t_data, x_data, y_data), 1)) # ST x 3 # (전체 데이터 크기X3) 형태로 텐서로 변환
    eqns = torch.FloatTensor(np.concatenate((t_eqns, x_eqns, y_eqns), 1)) # ST x 3

    print(f"Number of Time Steps: {T}, Number of sample points: {num_samples} out of {N}")

    return variables, c_tensor_data, eqns, T_star, X_star, Y_star, C_star, U_star, V_star, P_star # 학습을 위한 variables, c_tensor_data, eqns와 평가를 위한 T_star, X_star, Y_star, C_star, U_star, V_star, P_star 반환 
