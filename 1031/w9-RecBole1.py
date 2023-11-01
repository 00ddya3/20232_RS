# Created or modified on Oct 2023
# Author: 임일
# RecBole 1 --- 엄청 오래 걸림

from recbole.quick_start import run_recbole
run_recbole(model='ItemKNN', dataset='ml-100k')

# LightGCN 사용
run_recbole(model='LightGCN', dataset='ml-100k')

# CDAE(Collaborative Denosing Auto-Encoders) 사용
run_recbole(model='CDAE', dataset='ml-100k')

# ENMF(Efficient Neural Matrix Factorization) 사용
run_recbole(model='ENMF', dataset='ml-100k')

# FM(Factorization Machines) 사용
run_recbole(model='FM', dataset='ml-100k')

