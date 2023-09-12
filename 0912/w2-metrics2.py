# Created or modifed on Sep 2023
# @author: 임일
# Metrics

def b_metrics1(target, pred):       # 실제, 예측 item을 리스트로 받아서 precision, recall, F1 계산하는 함수
    n_target = len(target)          # item 개수 초기화
    n_pred = len(pred)
    n_correct = len(set(target).intersection(set(pred)))
    try:                            # 에러(division by zero 등)가 발생하는 경우를 대비해서
        precision = n_correct / n_pred
        recall = n_correct / n_target
        if (precision == 0 and recall == 0):  # Prevent 'division by zero'
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        return precision, recall, f1
    except:
        return 'error'

targets = [1, 100, 40, 20, 44]
predictions = [2, 100, 30, 44]
b_metrics1(targets, predictions)

def b_metrics2(target, pred, k=0):  # 실제, 예측 item과 k를 받아서 precision, recall, f1계산
    n_target = len(target)          # item 개수 초기화
    n_pred = len(pred)
    n_correct = len(set(target).intersection(set(pred)))
    if k > 0:                       # @k
        if k > n_pred:              # k가 예측 item 수보다 작은 경우 처리
            k = n_pred
        pred_k = pred[:k]
        n_correct_k = len(set(target).intersection(set(pred_k)))
    try:                            # 에러가 발생하는 경우를 대비해서
        precision = n_correct / n_pred
        recall = n_correct / n_target
        if (precision == 0 and recall == 0):  # Prevent 'division by zero'
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        if k > 0:
            precision_k = n_correct_k / k
            recall_k = n_correct_k / n_target
            return precision, recall, f1, precision_k, recall_k, k
        else:
            return precision, recall, f1
    except:
        return 'error'

targets = [1, 100, 40, 20, 44]
predictions = [2, 100, 30, 44]
b_metrics2(targets, predictions, 3)

def b_metrics(target, pred):        # 실습과제 w2-1 
    b_metrics1()
    return 

targets = [[1, 100, 40, 20, 44, 98], [3, 4, 10, 7]]
predictions = [[2, 100, 40, 30, 44], [4, 1, 3, 100]]
b_metrics(targets, predictions)








