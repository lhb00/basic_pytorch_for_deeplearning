# 라이브러리 불러오기 
import argparse # argparse 이용, 파이썬 안열고 터미널에서 실행 
from mat_to_numpy import load_data # 미리 만들어 놓은 모듈 불러옴 
import utils
import torch
import numpy as np
from time import time # 시간 확인용 함수 time 

# 기본값 설정 
if __name__ == "__main__": # 모든 코드는 main 조건 내부에서 작성 
    parser = argparse.ArgumentParser(description='Hidden Fluid Mechanics - Training') # parser 선언, 간단한 코드 설명 기입 
    parser.add_argument('--version_name', default='0', type=str, help='version name') # 학습에 필요한 기본값 설정, version_name은 파일 저장 시 버전에 따라 파일명을 다르게 하기 위한 문자열. 
    parser.add_argument('--datapath', default='/content/drive/MyDrive/deeplearning/HFM/data/Cylinder2D_flower.mat', type=str, help='data path')
    parser.add_argument('--modelpath', default=None, type=str, help='pretrained model path')
    parser.add_argument('--num_samples', default=100000, type=int, help='number of samples: N out of 157879')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size')
    parser.add_argument('--total_time', default=40, type=int, help='runtime') # total_time에서의 40은 40시간을 의미 
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    args = parser.parse_args()
    print(args)

    # Data
    # 데이터 불러오기
    # mat_to_numpy 모듈에 있는 load_data 이용, 데이터 불러옴. 
    data, c_data, eqns, T_star, X_star, Y_star, C_star, U_star, V_star, P_star = load_data(args.datapath,
                                                                                           args.num_samples)

    # Model
    # 모델 불러오기 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layer_list = [3] + 10 * [200] + [4] # 레이어의 노드 수를 가지고 있는 리스트 생성
    # 이 예시는 [3,200, 200, ..., 200, 4]로 구성, 모델 생성시 (입력노드, 출력노드) = (3,200), (200,200), ..., (200,200), (200,4)로 정보가 들어가 입력층 1개, 은닉층 8개, 출력층 1개를 갖는 다층 신경망 구축. 
    model = utils.pinn(data, layer_list) # 데이터 정규화를 위해 데이터를 입력하고 모델 구조를 결정 짓는 layer_list 입력, 모델 생성 

    if args.modelpath != None: # If 학습된 모델을 사용하기 위해 모델 파일 경로 args.modelpath 입력 시 학습된 모델 파라미터 불러옴. 
        model.load_state_dict(torch.load(args.modelpath))

    # Optimizer
    # 최적화 정의
    # 최적화 방법 : Adam, 주어진 학습률 args.lr 적용 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 학습 준비 
    start_time = time() # 학습 직전의 시간 체크 
    running_time = 0 # 학습 누적 시간 정의 
    it = 0 # 학습 반복 수 정의 
    min_loss = 1 # 모델 저장을 위해 최소 손실값 정의 

    print("Start training the model..")
    # 모델 학습 
    while running_time < args.total_time: # 설정한 최대 시간 args.total_time까지 학습 진행. 

        # batch data
        optimizer.zero_grad()
        idx_data = np.random.choice(args.num_samples, args.batch_size) # 배치 학습을 위해 데이터 샘플 추출 
        idx_eqns = np.random.choice(args.num_samples, args.batch_size)
        data_batch = data[idx_data, :].to(device)
        c_data_batch = c_data[idx_data, :].to(device)
        eqns_batch = data[idx_eqns, :].to(device)
        data_batch.requires_grad = True # 자동 미분을 위해 해당 데이터의 requires_grad 비활성화 
        c_data_batch.requires_grad = True
        eqns_batch.requires_grad = True

        # prediction
        data_outputs = model(data_batch) # 모델을 통해 c, u, v, p값 출력 
        c_data_pred = data_outputs[:, 0:1] # data_outputs의 0, 1, 2, 3번째 열은 c, u, v, p이다. => c_data_pred=data_outputs[:, 0:1]이다 [:,0:1]인 이유는 2차원 형태 텐서로 만들라고. 

        eqns_outputs = model(eqns_batch) # 똑같이 c, u, v, p값 예측 & 목적 함수 e1~e4 계산.
        c_eqns_pred = eqns_outputs[:, 0:1]
        u_eqns_pred = eqns_outputs[:, 1:2]
        v_eqns_pred = eqns_outputs[:, 2:3]
        p_eqns_pred = eqns_outputs[:, 3:4]

        e1, e2, e3, e4 = utils.Navier_Stokes_2D(c_eqns_pred, u_eqns_pred, v_eqns_pred, p_eqns_pred, eqns_batch, 100,
                                                100)

        # loss
        loss_c = torch.mean((c_data_pred - c_data_batch) ** 2) # 첫번째 예측에서 나온 c 이용, 손실 함수 loss_c 정의. 
        loss_e = torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2) + torch.mean(e4 ** 2) # 두 번째 예측에서 얻어진 e1~e4의 손실 함수 : loss_e라고 정의 
        loss = loss_c + loss_e # 최종적으로 두 손실 함수 더해 최적화의 기준이 되는 loss 정의 
        loss.backward() # 다음 최적화 진행 
        optimizer.step()

        if loss.item() < min_loss: # 첫 번째 if문에서는 손실값을 기준으로 모델 파라미터 저장 
            min_loss = loss.item()
            torch.save(model.state_dict(), './hfm_'+ args.version_name + '.pth')
            # print(f"It: {it} - Save the best model, loss: {loss.item()}")

        if it % 100 == 0: # 100회 학습마다 런타임 확인 & 진행 상황 출력 
            elapsed = time() - start_time
            running_time += elapsed / 3600.0
            print('Iteration: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh' % (it, loss, elapsed, running_time))
            start_time = time()

        if (it % 1000 == 0) and (it != 0): # 1000회 학습마다 임의로 특정 시간 데이터 추출 => 예측값 추출 
            # Prediction
            with torch.no_grad():
                variables_star, target_star = utils.test_data(T_star, X_star, Y_star, C_star, U_star, V_star, P_star)
                data_star_outputs = model(variables_star.to(device))
                c_star_pred = data_star_outputs[:, 0:1] # 각 변수로 나눠 데이터 정리 
                u_star_pred = data_star_outputs[:, 1:2]
                v_star_pred = data_star_outputs[:, 2:3]
                p_star_pred = data_star_outputs[:, 3:4]

                # Target (actual values)
                c_target = target_star[:, 0:1].to(device) # 마찬가지로 대응되는 실제값 정의 
                u_target = target_star[:, 1:2].to(device)
                v_target = target_star[:, 2:3].to(device)
                p_target = target_star[:, 3:4].to(device)

                c_error = utils.relative_error(c_star_pred, c_target) # 각 변수들의 상대오차 구해 평가 진행 
                u_error = utils.relative_error(u_star_pred, u_target)
                v_error = utils.relative_error(v_star_pred, v_target)
                p_error = utils.relative_error(p_star_pred, p_target)
                print('Error: c: %.3f, u: %.3f, v: %.3f, p: %.3f' % (c_error, u_error, v_error, p_error)) # 매 학습마다 학습 횟수 세고 만일에 대비, 학습마다 모델 저장 

        it += 1
        torch.save(model.state_dict(), './hfm_'+ args.version_name + '_last.pth')





