#!/bin/bash

# python gnn_nobatch.py --city beijing
# python gnn_nobatch.py --city shanghai
# python gnn_nobatch.py --city shenzhen

# python gnn_knngraph.py --city beijing --model gcn
# python gnn_knngraph.py --city shanghai --model gcn
# python gnn_knngraph.py --city shenzhen --model gcn

# python gnn_knngraph.py --city beijing --model graphsage
# python gnn_knngraph.py --city shanghai --model graphsage
# python gnn_knngraph.py --city shenzhen --model graphsage

# python gnn_knngraph.py --city beijing --model gat
# python gnn_knngraph.py --city shanghai --model gat
# python gnn_knngraph.py --city shenzhen --model gat

# python gnn_knngraph.py --city beijing --model gin
# python gnn_knngraph.py --city shanghai --model gin
# python gnn_knngraph.py --city shenzhen --model gin



k_list=(5 8 10 12 15 18 20 35 30 40 50 60)
# model_list = ('gcn' 'graphsage' 'gat' 'gin')
for k in ${k_list[*]}
    do
    echo "python knn_graph.py --n_neighbors ${k}"
    python knn_graph.py --n_neighbors ${k}
    wait
    python gnn_knngraph.py --model gcn --k ${k}
    python gnn_knngraph.py --model graphsage --k ${k}
    python gnn_knngraph.py --model gat --k ${k}
    python gnn_knngraph.py --model gin --k ${k}
    done
    
# echo "OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u ./pretrain.py --test_dataset ${test_dataset} --data_list $data_list --test_dataset ${test_dataset} --gpu ${gpu} > ./out/pretrain/${data_list}.out 2>&1 &"
# OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u ./pretrain.py --test_dataset ${test_dataset} --data_list $data_list --test_dataset ${test_dataset} --gpu ${gpu} > ./out/pretrain/${data_list}.out 2>&1 &
# wait
# mkdir ./out/pattern/${data_list}
# echo "OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u ./patch_devide.py --data_list $data_list --test_dataset ${test_dataset} > ./out/pattern/patch_devide_${data_list}.out 2>&1 &"
# OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u ./patch_devide.py --data_list $data_list --test_dataset ${test_dataset} > ./out/pattern/patch_devide_${data_list}.out 2>&1 &
# wait

# for K in ${Ks[*]}
# do
#     # echo "${test_dataset}" "${data_list}"
#     # mkdir ./out/pattern/${data_list}
#     # echo "OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u ./pattern_clustering.py --data_list $data_list --test_dataset ${test_dataset} --sim ${sim} --K ${K} > ./out/pattern/${data_list}/clustering_${sim}_${K}.out 2>&1 &"
#     # OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u ./pattern_clustering.py --data_list $data_list --sim ${sim} --K ${K} > ./out/pattern/${data_list}/clustering_${sim}_${K}.out 2>&1 &
#     # wait
#     # echo "OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u ./pattern_visualize.py --data_list $data_list --test_dataset ${test_dataset} --sim ${sim} --K ${K} > ./out/pattern/${data_list}/visualize_${sim}_${K}.out 2>&1 &"
#     # OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u ./pattern_visualize.py --data_list $data_list --sim ${sim} --K ${K} > ./out/pattern/${data_list}/visualize_${sim}_${K}.out 2>&1 &
#     # wait
#     for finetune_epochs in ${finetune_epochs_list[*]}
#     do
#         for update_step in ${update_step_list[*]}
#         do
#             for train_epochs in ${train_epochs_list[*]}
#             do
#                 for seed in ${seeds[*]}    
#                 do
#                 echo "OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u  main_maml_${modelprefix}.py --seed ${seed} --train_epochs ${train_epochs} --finetune_epochs ${finetune_epochs} --update_step ${update_step} --lr ${lr} --patch_encoder $patch_encoder --gpu $gpu --K ${K} --data_list ${data_list} --test_dataset ${test_dataset} > ${out_dir}${modelprefix}${patch_encoder}_${STmodel}_train${train_epochs}epochs_finetune${finetune_epochs}epochs_update${update_step}step_lr${lr}_K${K}_${seed}.out 2>&1 &"

#                 OMP_NUM_THREADS=${thread} CUDA_VISIBLE_DEVICES=${gpu} nohup python -u main_maml_${modelprefix}.py --seed ${seed} --train_epochs ${train_epochs} --finetune_epochs ${finetune_epochs} --update_step ${update_step} --lr ${lr}  --patch_encoder $patch_encoder --gpu $gpu --K ${K} --data_list ${data_list} --test_dataset ${test_dataset} > ${out_dir}${modelprefix}${patch_encoder}_${STmodel}_train${train_epochs}epochs_finetune${finetune_epochs}epochs_update${update_step}step_lr${lr}_K${K}_${seed}.out 2>&1 &
#                 wait
#                 done
#             done
#         done
#     done
# done