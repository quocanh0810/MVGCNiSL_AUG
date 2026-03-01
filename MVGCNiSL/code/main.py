import numpy as np
import pandas as pd
import argparse, sys, json, random
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt

from utils import *
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
from model import *
from datetime import datetime

import os
import glob

# # Cố định random seed
# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

def init_argparse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--type', type=str, default="feature", help="one of these: feature, comparison, test")

    parser.add_argument('--data_source', type=str, default="Jurkat", help="which cell line to train and predict")
    #K562, Jurkat

    parser.add_argument('--threshold', type=float, default=-3, help="threshold of SL determination")
    parser.add_argument('--specific_graph', type=lambda s:[item for item in s.split("%") if item != ""], default=["SL"], help="lists of cell-specific graphs to use.")
    parser.add_argument('--indep_graph', type=lambda s:[item for item in s.split("%") if item != ""], 
                    default=['PPI-genetic','PPI-physical','co-exp','co-ess'], help="lists of cell-independent graphs to use.")
    #['PPI-genetic','PPI-physical','co-exp','co-ess']
    parser.add_argument('--node_feats', type=str, default="precomputed_aug", help="gene node features")
    # precomputed, raw_omics, precomputed_aug

    parser.add_argument('--balanced', type=int, default=1, help="whether the negative and positive samples are balanced")
    parser.add_argument('--pos_weight', type=float, default=50, help="weight for positive samples in loss function")
    parser.add_argument('--CCLE', type=int, default=0, help="whether or not include CCLE features into node features")
    parser.add_argument('--CCLE_dim', type=int, default=64, help="dimension of embeddings for each type of CCLE omics data")
    parser.add_argument('--node2vec_feats', type=int, default=0, help="whether or not using node2vec embeddings")

    parser.add_argument('--model', type=str, default="GCN_pool", help="model type")
    parser.add_argument('--pooling', type=str, default="max", help="type of pooling operations")
    parser.add_argument('--LR', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--epochs', type=int, default=1500, help="number of maximum training epochs")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size")
    parser.add_argument('--out_channels', type=int, default=64, help="dimension of output channels")
    parser.add_argument('--patience', type=int, default=250, help="patience in early stopping")
    parser.add_argument('--training_percent', type=float, default=0.70, help="proportion of the SL data as training set")
    
    parser.add_argument('--save_results', type=int, default=1, help="whether to save test results into json")
    parser.add_argument('--split_method', type=str, default="novel_pair", help="how to split data into train, val and test")
    parser.add_argument('--predict_novel_genes', type=int, default=0, help="whether to predict on novel out of samples")
    parser.add_argument('--novel_cellline', type=str, default="Jurkat", help="name of novel celllines")

    args = parser.parse_args()

    return args


def train_model(model, optimizer, data, device, train_pos_edge_index, train_neg_edge_index):
    model.train()
    optimizer.zero_grad()
    x = data.x.to(device)
    edge_index_list = []
    for edge_index in data.edge_index_list:
        edge_index = edge_index.to(device)
        edge_index_list.append(edge_index)

    # shuffle training edges and labels
    all_edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], dim=-1)
    labels = get_link_labels(train_pos_edge_index, train_neg_edge_index, device)
    num_samples = all_edge_index.shape[1]
    all_idx = list(range(num_samples))
    np.random.shuffle(all_idx)
    all_edge_index = all_edge_index[:,all_idx]
    labels = labels[all_idx]

    start = 0
    loss = 0
    while start < num_samples:
        temp_z_list = []
        for edge_index in edge_index_list:
            temp_z = model.encode(x, edge_index)
            temp_z_list.append(temp_z)
        
        z = torch.cat(temp_z_list,1)
        # transpose is used to transform the data from (batch, # graphs, # features) into (batch, # features, # graphs)
        # the pooling operation is performed on the third dimension (graphs)
        z = z.unsqueeze(1).reshape(z.shape[0],len(edge_index_list),-1).transpose(1,2)
        
        if args.pooling == "max":
            z = F.max_pool2d(z, (1,len(edge_index_list))).squeeze(2)
        elif args.pooling == "mean":
            z = F.avg_pool2d(z, (1,len(edge_index_list))).squeeze(2)

        link_logits = model.decode(z, all_edge_index[:,start:(start+args.batch_size)])
        #link_probs = link_logits.sigmoid()
        link_labels = labels[start:(start+args.batch_size)]

        if args.balanced:
            pos_weight = torch.tensor(1)
        else:
            pos_weight = torch.tensor(args.pos_weight)

        batch_loss = F.binary_cross_entropy_with_logits(link_logits, link_labels, pos_weight=pos_weight)
        batch_loss.backward()
        optimizer.step()

        loss += batch_loss.item()
        start += args.batch_size

    return float(loss)


@torch.no_grad()
def test_model(model, optimizer, data, device, pos_edge_index, neg_edge_index):
    model.eval()
    results = {}
    x = data.x.to(device)

    all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index, device)

    temp_z_list = []
    for i, edge_index in enumerate(data.edge_index_list):
        edge_index = edge_index.to(device)
        temp_z = model.encode(x, edge_index)
        temp_z_list.append(temp_z)
    
    z = torch.cat(temp_z_list,1)
    z = z.unsqueeze(1).reshape(z.shape[0],len(data.edge_index_list),-1).transpose(1,2)
    
    if args.pooling == "max":
        z = F.max_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    elif args.pooling == "mean":
        z = F.avg_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    
    link_logits = model.decode(z, all_edge_index)
    link_probs = link_logits.sigmoid()

    if args.balanced:
        pos_weight = torch.tensor(1)
    else:
        pos_weight = torch.tensor(args.pos_weight)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels, pos_weight=pos_weight)

    results = evaluate_performance(link_labels.cpu().numpy(), link_probs.cpu().numpy())

    return float(loss), results


@torch.no_grad()
def predict_oos(model, optimizer, data, device, pos_edge_index, neg_edge_index):
    model.eval()
    x = data.x.to(device)

    temp_z_list = []
    for i, edge_index in enumerate(data.edge_index_list):
        edge_index = edge_index.to(device)
        if args.model == 'GCN_multi' or args.model == 'GAT_multi':
            i = torch.tensor(i).to(device)
            temp_z = model.encode(x, edge_index, i)
        else:
            temp_z = model.encode(x, edge_index)
        temp_z_list.append(temp_z)
    
    z = torch.cat(temp_z_list,1)
    z = z.unsqueeze(1).reshape(z.shape[0],len(data.edge_index_list),-1).transpose(1,2)

    if args.pooling == "max":
        z = F.max_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    elif args.pooling == "mean":
        z = F.avg_pool2d(z, (1,len(data.edge_index_list))).squeeze(2)
    
    # due to the huge size of the input data, split them into 100 batches
    batch_num = 100
    step_size_neg = int(neg_edge_index.shape[1]/batch_num) + 1
    link_probs = []
    for j in tqdm(range(batch_num)):
        temp_link_logits = model.decode(z, pos_edge_index, neg_edge_index[:,(j*step_size_neg):((j+1)*step_size_neg)])
        temp_link_probs = temp_link_logits.sigmoid()
        link_probs.extend(temp_link_probs.cpu().numpy().tolist())

    return link_probs


if __name__ == "__main__":
    args = init_argparse()
    print(args)
    graph_input = args.specific_graph + args.indep_graph
    print("Number of input graphs: {}".format(len(graph_input)))
    if len(graph_input) == 0:
        print("Please specify input graph features...")
        sys.exit(0)
    # load data
    data, SL_data_train, SL_data_val, SL_data_test, SL_data_oos, gene_mapping = generate_torch_geo_data(args.data_source, args.CCLE, args.CCLE_dim, args.node2vec_feats, 
                                    args.threshold, graph_input, args.node_feats, args.split_method, args.predict_novel_genes, args.training_percent)

    num_features = data.x.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if args.model == "GCN_pool":
        model = GCN_pool(num_features, args.out_channels, len(data.edge_index_list)).to(device)
    elif args.model == 'GCN_conv':
        model = GCN_conv(num_features, args.out_channels, len(data.edge_index_list)).to(device)
    elif args.model == 'GCN_multi':
        model = GCN_multi(num_features, args.out_channels, len(data.edge_index_list)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.LR)

    # generate SL torch data
    #train_pos_edge_index, train_neg_edge_index = generate_torch_edges(SL_data_train, args.balanced, True, device)
    val_pos_edge_index, val_neg_edge_index = generate_torch_edges(SL_data_val, True, False, device)
    test_pos_edge_index, test_neg_edge_index = generate_torch_edges(SL_data_test, True, False, device)
    if args.predict_novel_genes:
        oos_pos_edge_index, oos_neg_edge_index = generate_torch_edges(SL_data_oos, False, False, device)

    
    train_losses = []
    valid_losses = []
    # initialize the early_stopping object
    random_key = random.randint(1,100000000)
     # Kiểm tra và tạo thư mục nếu chưa tồn tại
    checkpoint_dir = "checkpoint"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = "checkpoint/{}.pt".format(str(random_key))
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, reverse=True, path=checkpoint_path)

    for epoch in range(1, args.epochs + 1):
        # in each epoch, using different negative samples
        train_pos_edge_index, train_neg_edge_index = generate_torch_edges(SL_data_train, args.balanced, True, device)
        train_loss = train_model(model, optimizer, data, device, train_pos_edge_index, train_neg_edge_index)
        train_losses.append(train_loss)
        val_loss, results = test_model(model, optimizer, data, device, val_pos_edge_index, val_neg_edge_index)
        valid_losses.append(val_loss)
        print('Epoch: {:03d}, loss: {:.4f}, AUC: {:.4f}, AP: {:.4f}, val_loss: {:.4f}, precision@5: {:.4f}, precision@10: {:.4f}'.format(epoch, 
                                        train_loss, results['AUC'], results['AUPR'], val_loss, results['precision@5'],results['precision@10']))
        
        #early_stopping(results['aupr'], model)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early Stopping!!!")
            break
    

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(checkpoint_path))

    test_loss, results = test_model(model, optimizer, data, device, test_pos_edge_index, test_neg_edge_index)
    print("\ntest result:")
    print('AUC: {:.4f}, AP: {:.4f}, precision@5: {:.4f}, precision@10: {:.4f}'.format(results['AUC'], results['AUPR'], 
                                                                                        results['precision@5'], results['precision@10']))
    
    # ----- TÍNH TOÁN TOP CẶP GENE CÓ ERROR THẤP NHẤT -----
    print("\nCalculating lowest error pairs...")

    test_edge_index = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=-1)
    test_labels = get_link_labels(test_pos_edge_index, test_neg_edge_index, device)

    # Tính toán embeddings
    model.eval()
    with torch.no_grad():
        temp_z_list = []
        for edge_index in data.edge_index_list:
            edge_index = edge_index.to(device)
            temp_z = model.encode(data.x.to(device), edge_index)
            temp_z_list.append(temp_z)

        z = torch.cat(temp_z_list, 1)
        z = z.unsqueeze(1).reshape(z.shape[0], len(data.edge_index_list), -1).transpose(1, 2)

        if args.pooling == "max":
            z = F.max_pool2d(z, (1, len(data.edge_index_list))).squeeze(2)
        elif args.pooling == "mean":
            z = F.avg_pool2d(z, (1, len(data.edge_index_list))).squeeze(2)

        link_logits = model.decode(z, test_edge_index)
        link_probs = link_logits.sigmoid()

    errors = torch.abs(test_labels - link_probs).cpu().numpy()
    gene1_idx = test_edge_index[0].cpu().numpy()
    gene2_idx = test_edge_index[1].cpu().numpy()

    idx_to_gene = {v: k for k, v in gene_mapping.items()}
    gene_pairs = [(idx_to_gene[gene1_idx[i]], idx_to_gene[gene2_idx[i]], errors[i]) for i in range(len(errors))]
    sorted_gene_pairs = sorted(gene_pairs, key=lambda x: x[2])

    print("\nTop 10 gene pairs with lowest error:")
    for i in range(10):
        gene1, gene2, error = sorted_gene_pairs[i]
        print(f"{i+1}. {gene1} - {gene2}: Error = {error:.6f}")

    # Kết thúc tính toán

    save_dict = {**vars(args), **results, "lowest_error_pairs": sorted_gene_pairs[:10]}
    
    
    if args.predict_novel_genes:
        print("Predicting on novel genes...")
        oos_preds = predict_oos(model, optimizer, data, device, oos_pos_edge_index, oos_neg_edge_index)
        save_dict['gene_mapping'] = gene_mapping
        save_dict['oos_samples_1'] = SL_data_oos['gene1'].values.tolist()
        save_dict['oos_samples_2'] = SL_data_oos['gene2'].values.tolist()
        save_dict['oos_pred'] = oos_preds

    def convert_to_serializable(obj):
        """
        Chuyển đổi mọi kiểu dữ liệu không tương thích JSON (numpy, torch, float32, tensor, dict, list)
        thành các kiểu dữ liệu gốc của Python để có thể lưu vào JSON.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Chuyển numpy array thành list
        elif isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.cpu().numpy().tolist()  # Tensor 1 phần tử -> float, mảng -> list
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()  # Chuyển numpy float & int thành kiểu Python gốc
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}  # Xử lý đệ quy với dict
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]  # Xử lý đệ quy với list
        elif isinstance(obj, torch.dtype):  # Kiểm tra nếu là kiểu dữ liệu torch
            return str(obj)  # Chuyển thành string
        elif isinstance(obj, (int, float, str, bool)):  # Kiểu dữ liệu Python gốc
            return obj
        else:
            return str(obj)  # Chuyển bất kỳ thứ gì khác thành chuỗi để tránh lỗi


    # Chuyển đổi toàn bộ `save_dict` sang kiểu có thể lưu vào JSON
    save_dict = convert_to_serializable(save_dict)

    # Lấy ngày giờ hiện tại và định dạng thành chuỗi "YYYYMMDD_HHMMSS"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_results = pd.DataFrame(sorted_gene_pairs, columns=["Gene1", "Gene2", "Error"])
    df_results.to_csv(f"../results_genpairs/gene_pairs_{args.data_source}_{args.split_method}_{current_time}.csv", index=False)

    print(f"✅ Kết quả cặp gene đã được lưu vào: gene_pairs_{args.data_source}_{args.split_method}_{current_time}.csv")

    # Tạo tên file mới với format ngày tháng năm
    file_name = "MVGCN_{}_{}_{}_{}.json".format(args.data_source, args.split_method, current_time, str(random_key))

    # Kiểm tra và tạo thư mục nếu chưa tồn tại
    results_dir = "../results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    if args.save_results:
        with open(os.path.join(results_dir, file_name), "w") as f:
             json.dump(save_dict, f, indent=4)

    print(f"✅ Kết quả đã được lưu vào: {file_name}")



    # Đọc tất cả các file gene_pairs
    result_files = glob.glob("../results_genpairs/Jurkat_40/gene_pairs_*.csv")

    if len(result_files) > 1:
        all_data = []
    
    # Đọc tất cả các file và lưu vào list
    for file in result_files:
        df = pd.read_csv(file)
        all_data.append(df)

    # Gộp tất cả dữ liệu thành một DataFrame
    df_all = pd.concat(all_data)

    # Đảm bảo rằng mỗi cặp gene có cột 'error' tính toán được
    # Tính số lần mỗi cặp gene xuất hiện
    stability_count = df_all.groupby(["Gene1", "Gene2"]).size().reset_index(name="Count")

    # Tính error trung bình cho mỗi cặp gene
    avg_error = df_all.groupby(["Gene1", "Gene2"])["Error"].mean().reset_index(name="AvgError")

    # Gộp thông tin "Count" và "AvgError"
    stability_count = pd.merge(stability_count, avg_error, on=["Gene1", "Gene2"])

    # Lọc ra cặp gene xuất hiện > 50% số lần chạy và có error thấp hơn ngưỡng (ví dụ: 0.2)
    stable_pairs = stability_count[(stability_count["Count"] > len(result_files) * 0.5) & 
                                   (stability_count["AvgError"] < 0.15)]

    print(f"📊 Tổng số file kiểm tra: {len(result_files)}")
    print(f"🔍 Số cặp gene xuất hiện trong > 50% lần chạy và có error thấp hơn 0.025: {len(stable_pairs)}")
    print(stable_pairs.head(10))  # In ra 10 cặp ổn định nhất có error thấp

    # Lưu lại danh sách các cặp ổn định có error thấp
    stable_pairs.to_csv("../results_genpairs_stable/stable_gene_pairs_with_low_error.csv", index=False)
    print(f"✅ Danh sách cặp gene ổn định có error thấp đã được lưu vào: stable_gene_pairs_with_low_error.csv")
